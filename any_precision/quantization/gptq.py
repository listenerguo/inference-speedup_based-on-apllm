# Adapted from AutoGPTQ (https:XXXX) and modified by XX LXX


import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer_gptq import Quantizer


logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import matplotlib.pyplot as plt
import numpy as np
import os

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class GPTQ:
    def __init__(self, layer, module_name, module_type):
        self.layer = layer  # 这里的layer 本质上指的是 具体的modules，也就是最小子模块，如 线性层；而非decoderlayer，复合模块,如 注意力层，前馈网络层
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)    # 从1维展开： W[2,3,4]->[2，3*4]
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer(module_name, module_type)
        # print(f'***// 整形后 权重矩阵大小: {W.shape} //***')
        # print(f'***// 海森矩阵大小: {self.H.shape} //***')

        self.act_stat = "ea2"# 将"rms"修改为 二阶矩(ea2)  # 统计指标：默认rms，可改为"max"/"mean"
        self.act_channel = torch.zeros(self.columns, device=self.dev)  # 存储通道级统计结果
        self.act_count = 0  # 累计“有效样本数”（按输入列数计算，适配卷积层）

        self.name = module_name
        self.module_type = module_type
        self.grad = None

    def add_batch(self, inp, out):
        # 计算并更新海森矩阵，同时确定激活权重因子
        # print(f'输入inp的形状{inp.shape}')   # [batch, seq_len, dim]

        # if os.environ.get("DEBUG"):
        #     self.inp1 = inp
        #     self.out1 = out

        if len(inp.shape) == 2: # 如果输入形状为[x,y], 则增加一个维度[1,x,y]
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]  # 记录当前batch数量
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))   # #  将前两维度合并， 将3维 inp 展为 2维[ ？,inp.shape[-1]]
            inp = inp.t()   # 转置    # 行：表示 特征维度； 列：表示 样本数量
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,         # 卷积核大小
                dilation=self.layer.dilation,   # 空洞卷积的扩充率
                padding=self.layer.padding,     # 赋零填充大小
                stride=self.layer.stride,       # 步长
            )
            inp = unfold(inp)   # 展平操作，将inp局部区域展开为列
            inp = inp.permute([1, 0, 2])    # 交换前两维的顺序  从 [batch, features, patches] 变为 [features, batch, patches]
            inp = inp.flatten(1)    # 从第1维，展平为行，即 将 [features, batch, patches] 变为 [features, batch*patches]
        # print(f'输入inp的形状 -- 展平后{inp.shape}')    # nn.Linear:  [dim, batch * seq_len]

        # # ============ Activation statistics online update ============
        # #  统计每个输入通道的激活幅值特征（默认 RMS），并做在线平均
        with torch.no_grad():
            # cur_feat: shape (channels,)
            if inp.numel() == 0:
                return
            if self.act_stat == "ea2":
                cur_feat =  torch.sqrt(torch.diag(inp @ (inp.t() @ inp) @ inp.t()))    #(inp ** 2).mean(dim=1).to(self.dev)  # 二阶矩
                # cur_feat = torch.sqrt((inp ** 2).mean(dim=1) + 1e-12).to(self.dev)    # 均方根：sqrt(mean(x^2))
            elif self.act_stat == "max":
                cur_feat = inp.abs().amax(dim=1).to(self.dev)
            elif self.act_stat == "mean":  # 'mean'
                cur_feat = inp.abs().mean(dim=1).to(self.dev)
            else:
                print("error")
            # cur_cols = inp.shape[1] # 当前“样本数量”按列数计算（Conv unfold 会把 patch 展开为更多列）
            if self.act_count == 0:  # 初始化 激活值的统计特征
                n = min(cur_feat.shape[0], self.act_channel.shape[0])
                # print(self.act_channel.shape, cur_feat.shape)
                self.act_channel[:n] = cur_feat[:n]
                self.act_count = 1
            else:  # 激活值 统计特征更新
                n = min(cur_feat.shape[0], self.act_channel.shape[0])
                # 仅更新前 n 个通道（保证 shape 对齐）
                self.act_channel[:n] = (
                        self.act_channel[:n] * (self.act_count / (self.act_count + 1))
                        + cur_feat[:n] * (1 / (self.act_count + 1))
                )
                self.act_count += 1

        # # ============ Hessian Calculation ============
        self.H *= self.nsamples / (self.nsamples + tmp) # 进行衰减加权   self.nsamples旧样本数 tmp新样本数
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()    # 进行缩放操作， 缩放因子为  math.sqrt(2 / self.nsamples)
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())   # 计算并更新海森矩阵的估计值 shape--out*out    H = H + inp * (inp^T)
        # print(f'***// 海森矩阵大小: {self.H.shape} //***')

        # print('权重维度：', self.rows, self.columns ,'\t特征维度：', inp.shape, '\t海森矩阵维度：',self.H.shape, '\t激活分布特征维度：',self.act_channel.shape)
        # W (out_f, input_f);   X (input_f, num_sample);    H (input_f, input_f)    A (input_f)

    def fasterquant(
        self,
        hessian_f,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,          # 在 static_groups 为真时，设置执行
        actorder=False,
        static_groups=False,    # 当其为True时，与group_size配合使用
        root_bit = 8,
        des_bit = 4,
    ):
        # print("now group_size_value:", group_size)
        # print(f"fasterquant函数输入参数 默认配置： group_size：{group_size}； actorder：{actorder}， static_groups：{static_groups}")
        W = self.layer.weight.data.clone()      # 模型权重
        # print(f"fasterquant函数输入参数： W：{W.shape}")
        if isinstance(self.layer, nn.Conv2d):
            # print('flatten')
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            # print('T')
            W = W.t()
        # print('! no operate')
        W = W.float()

        # H_diag = torch.diag(self.H).clone()
        tick = time.time()

        H = self.H
        del self.H
        # 如果海森矩阵对角元素存在0，则表示 对应权重 对 模型无影响(可将对应输入的模型权重设置为0)，同时，海森矩阵为0 会影响后续计算(重新设置为1)。
        # 这行代码首先调用 torch.diag(H) 来提取张量 H 的对角线元素，然后通过 == 0 比较操作，检查这些对角线元素是否等于0。
        # 如果对角线元素等于0，则对应的结果为 True；否则为 False。这样，dead 成为一个布尔型张量，其每个元素表示 H 的对角线上相应元素是否为0。
        dead = torch.diag(H) == 0   # 获取 H 对角线元素为0的 位置
        H[dead, dead] = 1   # 将所有对角元素为0 的 重新设置为 1
        W[:, dead] = 0      # 将权重W对应 列 置为 0

        para_h = hessian_f
        para_a = self.act_channel  # 激活值因子
        scale = []
        zero = []
        g_idx = []      # 返回值
        other_bit_scale = []
        other_bit_zero = []
        rbit_weight = []
        now_idx = 1
        error_c = []  # 记录补偿后 与真实权重间的差值


        # 计算全局量化参数
        # print('海森矩阵对角张量形状：',torch.diag(H).shape)    # [2048]
        if not self.quantizer.ready() and group_size == -1:  # 如果 ready() 为false， 则if语句为 True
            # print('为什么不 ready ？')
            self.quantizer.find_params(W, weight=True)  # 为整个W 计算 量化参数 - self.quantizer.zero 和 .scale
            e1_c = self.quantizer.find_comp(W, para_a, para_h, root_bit, des_bit, weight=True)
            error_c.append(e1_c)

        if static_groups:   # 默认情况下，不执行该if分支
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)

                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:    # 按照H矩阵对角线元素，为权重，决定量化顺序
            perm = torch.argsort(torch.diag(H), descending=True)    # perm = tensor([0, 3, 1, 2]) 表示 1列 放置 3列 的结果
            W = W[:, perm]  # 按照perm的顺序，按列排序
            H = H[perm][:, perm]    # 先行置换 H' = H[perm]； 再列置换 H = H'[:, perm]
            invperm = torch.argsort(perm)   # 最后还原 invperm = tensor([0, 2, 3, 1]) 表示 索引列 移动到 值列，1列移动到2列，2->3

        # Cholesky分解
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        # Q_int = torch.zeros_like(W)     # 存储量化后的 root_bit 整数权重

        damp = percdamp * torch.mean(torch.diag(H))  # percdamp * 对角元素 平均值
        diag = torch.arange(self.columns, device=self.dev)  # 生成一维张量，[0,1, ..., columns-1]
        H[diag, diag] += damp   # 等价于 H[diag[0],diag[0]] += damp、 H[diag[1],diag[1]] += damp  # 对角线加阻尼，避免奇异矩阵
        H = torch.linalg.cholesky(H)    # # 分解为下三角矩阵L
        H = torch.cholesky_inverse(H)   # # 求逆
        H = torch.linalg.cholesky(H, upper=True)    # # 再分解为上三角矩阵U
        Hinv = H    # # Hinv是H的近似逆矩阵，用于误差传播
        # H_diag = torch.diag(Hinv).clone()
        # print('Hessian matrix: ', Hinv, Hinv.shape)
        # print('Module size:', W.shape)

        # 分块量化
        for i1 in range(0, self.columns, blocksize):    # 按照blocksize 分块处理 [0, columns],步长 blocksize
            i2 = min(i1 + blocksize, self.columns)  #   # i2-当前block上边界； 确保  i1 + blocksize 不超过 上限 self.columns
            count = i2 - i1 # 实际的块大小
            W1 = W[:, i1:i2].clone()    # 动态补偿权重 （后续会被 补偿）
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # 逐列量化
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:    # 如果设置了group_size
                    if not static_groups:   # 同时 static_group 为false
                        if (i1 + i) % group_size == 0:
                            # 计算量化参数 以及 完成量化精度升级
                            w_group = W[:, (i1 + i) : (i1 + i + group_size)]
                            # a_group = para_a[(i1 + i): (i1 + i + group_size)]
                            # h_group = para_h[:, (i1 + i): (i1 + i + group_size)]
                            self.quantizer.find_params(w_group, weight=True)
                            # e_c = self.quantizer.find_comp(w_group, a_group, h_group, root_bit, des_bit, weight=True)
                            e_c = self.quantizer.find_comp(w_group, 
                                                           para_a[(i1 + i): (i1 + i + group_size)], 
                                                           para_h[:, (i1 + i): (i1 + i + group_size)], 
                                                           root_bit, des_bit, weight=True)
                            error_c.append(e_c)

                        if ((i1 + i) // group_size) - now_idx == -1:    # x//y : 整数除法（向下取整）
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            other_bit_scale.append(self.quantizer.scales)
                            other_bit_zero.append(self.quantizer.zeros)
                            rbit_weight.append(self.quantizer.qw_rbit)
                            del self.quantizer.qw_rbit
                            del self.quantizer.scales
                            del self.quantizer.zeros
                            now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()  # q.shape=[2048,1] # 根据固定的量化参数和更新后的W 计算反量化 权重值q  # w 会更新
                Q1[:, i] = q
                # q_int = self.quantizer.quantize_int(w.unsqueeze(1)).flatten() # ！！后期实验 将w改为 W[:, ii+i]
                # # Q1_int[:, i] = q_int
                # # Q_int[:, i1+i] = q_int
                # 块内： 计算 量化误差 ，更新后续列权重，存储误差值
                Losses1[:, i] = (w - q) ** 2 / d**2     # d 属于 Hinv ，因此，d 越小，重要性越高
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))    # 块内误差传播
                Err1[:, i] = err1

            # 块间： 量化
            Q[:, i1:i2] = Q1    # 保存 根据补偿 W1 重新计算得到的 反量化结果 q 为 Q
            # Q_int[:, i1:i2] = Q1_int
            Losses[:, i1:i2] = Losses1 / 2      # # 汇总当前块的损失（除以2是为了后续平均）
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])  # 用当前块的误差修正后续未处理的块（跨块误差传播）

        # 后处理
        torch.cuda.synchronize()
        logger.info(f"[fp-int]duration: {(time.time() - tick)}")    # # 记录量化补偿总耗时
        logger.info(f"[fp-int]{self.name}-avg loss: {torch.sum(Losses).item() / self.nsamples}")    # # 计算并记录平均量化损失

        error_c = torch.stack(error_c).sum(dim=0)
        qbit_keys = list(range(des_bit, root_bit+1))  # int 逐级 量化范围
        for i, bit in enumerate(qbit_keys):
            logger.info(f"{bit}-bit avg loss: {(error_c[0][i])/ self.nsamples}")  # -error_noc[0][i] and {error_temp[0][i]/(row*column) / self.nsamples}")

        group_size = group_size if group_size != -1 else self.columns   # 如果此时group_size = -1 ，则其会被更新为 self.columns
        if static_groups and actorder:  # 如果两个同时激活(不执行该情况)
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:   # 一般情况下，执行该if 分支，// 即 两者都为false，或 任意一个为false
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]  # 属于 第几个 group
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        # 将 补偿后W计算出的Q，作为模型权重 (后续该层会)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        if scale == []: # 长度为1的列表，元素为张量[dim,1]   // 即 group_size = -1； static_groups = false;
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
            other_bit_scale.append(self.quantizer.scales)
            other_bit_zero.append(self.quantizer.zeros)
            rbit_weight.append(self.quantizer.qw_rbit)

        scale_4 = torch.cat(scale, dim=1)                 #  shape:[bits, outf, inf/gsize]
        scale_4 = scale_4.unsqueeze(0)
        scale_other = torch.stack(other_bit_scale, dim=0)
        scale_other = scale_other.squeeze(-1)
        scale_other = scale_other.permute(1, 2, 0)
        scales = torch.cat([scale_4, scale_other], dim=0)

        zero_4 = torch.cat(zero, dim=1)                 #  shape:[bits, outf, inf/gsize]
        zero_4 = zero_4.unsqueeze(0)
        zero_other = torch.stack(other_bit_zero, dim=0)
        zero_other = zero_other.squeeze(-1)
        zero_other = zero_other.permute(1, 2, 0)
        zeros = torch.cat([zero_4, zero_other], dim=0)
        # zero = torch.cat(zero, dim=1)                   #  shape:[outf, inf/gsize]

        rbit_weight = torch.cat(rbit_weight, dim=1)     # shape:[outf, inf]
        return  scales, zeros, g_idx, rbit_weight       # scales, zero, g_idx, rbit_weight     #scale, zero, g_idx, Q_int.to(torch.uint8), H_diag

    def quanterrorcomp(
            self,
            hessian,
            blocksize=128,
            group_size=-1,  # 在 static_groups 为真时，设置执行
            actorder=False,
            static_groups=False,  # 当其为True时，与group_size配合使用
            root_bit=8,
            des_bit=4,
    ):
        if actorder is True or static_groups is True:
            raise ValueError("该值为true的功能尚未开发")

        W = self.layer.weight.data.clone()  # 模型权重
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        H = hessian     # 接收的hession 本身就是基于方阵的得到的对角元素，一维
        A = self.act_channel  # 激活值因子
        print(f"重要参数规模: hessian: {H.shape}, activation:{A.shape}")

        c_comp = []  # 补偿常数
        now_idx = 1
        error_c = [] # 记录补偿后 与真实权重间的差值
        # error_noc= []  # 记录补偿前 差值

        tick = time.time()
        qbit_keys = list(range(root_bit - 1, des_bit - 1, -1))  # int 逐级 量化范围

        if not self.quantizer.ready():  # 如果 ready() 为false， 则if语句为 True
            e1_c, e1_noc = self.quantizer.find_comp(W, A, H, root_bit, des_bit, weight=True)    # 为整个W 计算 量化补偿参数 self.quantizer.comp

        # 分块量化
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            for i in range(count):

                if group_size != -1:  # 如果设置了group_size
                    if not static_groups:  # 同时 static_group 为false
                        if (i1 + i) % group_size == 0:
                            # self.quantizer.find_params(W[:, (i1 + i): (i1 + i + group_size)], weight=True)
                            e_c = self.quantizer.find_comp(W[:, (i1 + i): (i1 + i + group_size)],
                                                           A[(i1 + i): (i1 + i + group_size)],
                                                           H[:, (i1 + i): (i1 + i + group_size)],
                                                           root_bit, des_bit, weight=True)     # , e_noc
                            error_c.append(e_c)
                            # error_noc.append(e_noc)

                        if ((i1 + i) // group_size) - now_idx == -1:    # x//y : 整数除法（向下取整）
                            c_comp.append(self.quantizer.comp)
                            now_idx += 1

        torch.cuda.synchronize()
        logger.info(f"[int-int]duration: {(time.time() - tick)}")  # # 记录量化总耗时

        if error_c == []:
            error_c = e1_c
            # error_noc = e1_noc
        error_c = torch.stack(error_c).sum(dim=0)
        # error_noc = torch.stack(error_noc).sum(dim=0)
        # # print('/*-****NEW - NEW****/-', error_loss)
        row,column = W.shape
        sum_err1 = [0] * len(qbit_keys)
        # sum_err2 = [0] * len(qbit_keys)
        for i, bit in enumerate(qbit_keys):
            logger.info(f"{bit}-bit avg loss: {(error_c[0][i])/ self.nsamples}")  # -error_noc[0][i] and {error_temp[0][i]/(row*column) / self.nsamples}")
            sum_err1[i] = error_c[0][i]
            # sum_err2[i] = error_noc[0][i]

        if c_comp == []:
            c_comp.append(self.quantizer.comp)

        c_comp = torch.stack(c_comp, dim=0)
        c_comp = c_comp.permute(1, 2, 0)
        # c_comp = torch.cat(c_comp, dim=2)  # 第一维 为量化的bit， 第二 第三维 才是量化参数 --后两维与scale一致

        return c_comp, sum_err1 #, sum_err2


    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


    def plot_hessian(self, data=None, save_dir="hessian_plots"):
        """
        绘制海森矩阵对角线元素与激活值元素的双轴折线图，以及海森矩阵热力图（抽样）
        """
        # if self.H is None and data is None:
        #     logger.warning("海森矩阵H尚未计算，无法绘图")
        #     return
        if self.act_channel is None:
            logger.warning("激活值A尚未计算，无法绘图")
            return
        if not hasattr(self, "act_channel"):
            logger.warning("激活值数据act_channel不存在，跳过激活值绘图")
            return

        os.makedirs(save_dir, exist_ok=True)
        if data is None:
            H_np = self.H.cpu().detach().numpy()
            H_diag = np.diag(H_np)  # 海森矩阵对角线元素
        else:
            H_np = data.cpu().detach().numpy()
            H_diag = np.diag(H_np)
        A_np = self.act_channel.cpu().detach().numpy()  # 激活值元素

        # 1. 绘制双轴折线图（共享x轴，左右y轴分别表示H对角线和激活值）
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # 左侧y轴：海森矩阵对角线元素
        color = 'tab:blue'
        ax1.set_xlabel("权重索引")
        ax1.set_ylabel("Hessian对角线值（越大越重要）", color=color)
        ax1.plot(H_diag, color=color, label="Hessian对角线")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(alpha=0.3)

        # 创建右侧y轴：激活值元素
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel("激活值元素值（越大影响越强）", color=color)
        ax2.plot(A_np, color=color, linestyle='--', label="激活值")
        ax2.tick_params(axis='y', labelcolor=color)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f"{self.name} - Hessian对角线与激活值分布")
        combined_save_path = os.path.join(save_dir, f"{self.name}_combined.png")
        plt.savefig(combined_save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"双轴合并图已保存至: {combined_save_path}")

        if data is None:
            # 2. 绘制海森矩阵热力图（高维时抽样前100x100，避免内存溢出）
            sample_size = min(1024, H_np.shape[0])
            H_sample = H_np[100:sample_size+100, 100:sample_size+100]
            plt.figure(figsize=(8, 6))
            plt.imshow(H_sample, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="值")
            plt.title(f"{self.name} - 海森矩阵热力图（前{sample_size}x{sample_size}）")
            plt.xlabel("权重索引")
            plt.ylabel("权重索引")
            heatmap_save_path = os.path.join(save_dir, f"{self.name}_hessian_heatmap.png")
            plt.savefig(heatmap_save_path, bbox_inches="tight")
            plt.close()
            logger.info(f"海森矩阵热力图已保存至: {heatmap_save_path}")


        # os.makedirs(save_dir, exist_ok=True)
        # H_np = self.H.cpu().detach().numpy()  # 转移到CPU并转为numpy
        # H_diag = np.diag(H_np)  # 提取对角线元素（权重重要性）
        # # 1. 绘制对角线元素折线图（反映权重重要性分布）
        # plt.figure(figsize=(10, 4))
        # plt.plot(H_diag)
        # plt.title(f"{self.name} - 海森矩阵对角线元素（权重重要性）")
        # plt.xlabel("权重索引")
        # plt.ylabel("Hessian对角线值（越大越重要）")
        # plt.grid(alpha=0.3)
        # diag_save_path = os.path.join(save_dir, f"{self.name}_hessian_diag.png")
        # plt.savefig(diag_save_path, bbox_inches="tight")
        # plt.close()
        # logger.info(f"海森矩阵对角线图已保存至: {diag_save_path}")
        #
        # # 1.1  绘制 激活值折线图（反映权重重要性分布）
        # A_np = self.act_channel.cpu().detach().numpy()
        # plt.figure(figsize=(10, 4))
        # plt.plot(A_np)
        # plt.title(f"{self.name} - 激活值元素（权重被影响程度）")
        # plt.xlabel("权重索引")
        # plt.ylabel("激活值元素值（越大越重要）")
        # plt.grid(alpha=0.3)
        # diag_save_path = os.path.join(save_dir, f"{self.name}_activate_value.png")
        # plt.savefig(diag_save_path, bbox_inches="tight")
        # plt.close()
        # logger.info(f"激活值元素图已保存至: {diag_save_path}")

__all__ = ["GPTQ"]

# 计算误差
# error_rtod =
# w - (scale * (q - zero))    #  # 这里 w 可以尝试替换为 Q1[:, i] 但对比试验看效果 ！！！
# e_max = error_temp.max()    # 获取误差中的最大值元素
# e_min = error_temp.min()
# error = w - (q_subbit + c)

# error_c = w - (scale * (q_subbit + c) - zero) = w - (scale * q_subbit + scale * c -zero) = w - scale * q_subbit
# error_temp = w - (scale * (q_subbit - zero))
# 推导出 argmin_c sum( ir * error_c ) => 一阶导数为0 => c = sum( ir * (w-error_temp) ) / ( sacle * sum(ir))
# ik = 0.5 * ix + 0.5 * iw
#
# c = sum(ik * (w-q_subbit)) / (scale * sum(ik))
# # MSE-mesh search
# min_mes_c = inf # torch.full([x.shape[0]], float("inf"), device=dev)
# for i in range(100):
#     c = e_min + i * (e_max - e_min)/100
#     # 计算误差均方根最小的
#     error_c = w - (scale * (q_subbit - zero) + c)
#     # 将误差计算为 均方根形式
#     # 用最小的误差对应的c 替代
#     if error_c<min_mes_c:
#         min_mes_c = error_c