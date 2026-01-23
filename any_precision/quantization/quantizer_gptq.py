# Adapted from AutoGPTQ (https:XXXX) and modified by XX LXX

from logging import getLogger

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import os

import torch.nn.functional as F

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


logger = getLogger(__name__)


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)     # 量化操作(包含四舍五入)
    return scale * (q - zero)   # 反量化操作，返回的为根据量化后的整数，反量化转换 还原的浮点数。
    # print('量化后的整数',q)
    # print(f"输入形状{x.shape},  量化参数形状：{scale.shape},  量化结果形状{q.shape},  输出形状： {(scale * (q - zero)).flatten().shape}")

def quantize_int(x, scale, zero, maxq):
    # q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)     # 量化操作(包含四舍五入)
    zs = zero * scale
    q = torch.clamp(torch.round((x + zs) / scale).to(torch.int), 0, maxq)
    # q = torch.round((x + zs) / scale).to(torch.int)
    return q    # 返回 量化后的整数
    # print('对比量化权重结果：',q, int_w)
    # print(f"{x.shape}, {scale.shape}, {zero.shape}, {maxq}")



class Quantizer(nn.Module):
    def __init__(self, module_name, module_type, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))   # 最低量化/均匀量化时的 scale
        self.register_buffer("zero", torch.zeros(shape))    # 最低量化/均匀量化时的 zero
        self.register_buffer("scales", torch.zeros(shape))  # 多个比特的scales 集合
        self.register_buffer("zeros", torch.zeros(shape))   # 多个比特的zeros 集合
        self.register_buffer("qw_rbit", torch.zeros(shape))
        self.name = module_name
        self.module_type = module_type
        self.count = 0


    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse      # 通过对量化误差计算均方根，作为目标函数执行网格搜索，确定最佳量化参数
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        # 对输入x(通常为模型的权重矩阵)，计算 合适的量化 zero 和 scale
        dev = x.device
        self.maxq = self.maxq.to(dev)   # 获取 量化后对应bit下的最大值， 即2**bits - 1
        shape = x.shape

        # 将输入张量x，根据不同要求，转化为一个二维张量x
        if self.perchannel:
            if weight:      # 权重 张量处理
                x = x.flatten(1)    # 权重矩阵通常是 2D 的 [out_features, in_features]，此时 flatten(1) 前后无变化
            else:           # 激活 张量处理
                if len(shape) == 4: # 4维
                    x = x.permute([1, 0, 2, 3])     # [N,C,H,W] -> [C,N,H,W]
                    x = x.flatten(1)                # [C,N,H,W] -> [C,N*H*W]
                if len(shape) == 3: # 3维
                    x = x.reshape((-1, shape[-1])).t()  # [seq, batch, hidden] -> [hidden, seq*batch]
                if len(shape) == 2: # 2维
                    x = x.t()   # [batch, features] -> [features, batch]
        else:
            x = x.flatten().unsqueeze(0)    # # 所有值都在一个"通道"中进行全局量化 [1* y]

        tmp = torch.zeros(x.shape[0], device=dev)   # 全为0的x 张量， shape = (1, x.shape[0])
        xmin = torch.minimum(x.min(1)[0], tmp)      # 得到x每行的最小值(范围（负数到0）)  比较x在每行最小值元素（x.min(1)[0]） 与0 （tmp）的大小，并取最小值
        xmax = torch.maximum(x.max(1)[0], tmp)      # 得到x每行的最大值(范围（0到正数）)  比较 与 0 的大小，并取 最大值
        # print('!IMPORTANT',xmax.shape)    # [行]

        if self.sym:  # 对称量化 -- 将 最大值和最小值，分别调整为 绝对值最大 对应的正负值
            xmax = torch.maximum(torch.abs(xmin), xmax)     # xmax被更新为xmin的绝对值和xmax中的较大值
            tmp = xmin < 0  # 逐xmin的元素i 比较条件i<0, 是否成立，并返回True or False 的张量
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        # 处理全零数据（避免xmin 和 xmax 都为0）
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:   # 即表示 trits 为 True；(二值量化) 一般情况下，不执行该设置
            self.scale = xmax
            self.zero = xmin
        else:   # 标准的量化参数
            self.scale = (xmax - xmin) / self.maxq  # 量化范围为 [0, 2**bits-1] 假设4bit->[0000, 1111](二进制)->[0，15](十进制)
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)    # zero = 2**bits / 2
                # full_like(a, b)：将按照 a 的形状，生成一个值全为b的张量
            else:
                self.zero = torch.round(-xmin / self.scale)   # 期望 xmin 对应于 qmin; 即 qmin = (xmin - zero)*scale = 0

        if self.mse:    # TODO 参考该方法，设计补偿参数求解
            # 通过对权重范围进行收缩，计算量化参数，并通过量化误差最小化，寻找最合适的量化参数scale 和 zero
            best = torch.full([x.shape[0]], float("inf"), device=dev)   # 填充生成 张量形状为[行], 值为inf无穷大 的张量
            for i in range(int(self.maxshrink * self.grid)):    # 循环次数 maxshrink * grid （两个参数控制mse的收缩率与网格数）
                p = 1 - i / self.grid   # 收缩率
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        # 如果输入为激活值，对不同shape的处理
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def find_comp(self, w, a, h, rbit, dbit, weight=False):
        """
        Bit-Serial 量化求解器 (包含 Hessian 加权 MSE + Cosine Similarity 优化)

        Args:
            w: 原始权重 [Out, In]
            a: 激活 (本代码中未直接使用，假设集成在 H 中)
            h: Hessian 对角近似 [Out, In] (即 input code 中的 ir)
            rbit: 目标最高比特 (e.g. 8)
            dbit: 基础比特 (e.g. 4)
            weight: 是否为权重模式 (flatten 处理)
            lambda_cos: Cosine 损失的权重系数
        """

        if self.perchannel:
            if weight:  # 权重 张量处理
                w = w.flatten(1)  # 权重矩阵通常是 2D 的 [out_features, in_features]，此时 flatten(1) 前后无变化

        ir = h  # **0.8 * a**(1-0.8)
        # weight_ir_sum = torch.sum(ir, dim=1)  # 形状 [I]
        # 将输入张量w，根据不同要求，转化为一个二维张量x

        # 逐bit计算和分析
        curr_scale = self.scale.clone()
        curr_zero = self.zero.clone()
        all_errors = []
        all_scales = []
        all_zeros = []
        # dbit 4 比特量化结果
        q_curr = quantize_int(w, self.scale, self.zero, self.maxq)
        # 记录初始误差
        w_curr = self.scale * (q_curr - self.zero)
        dbit_error = (w - w_curr).abs().sum()  # 依然返回 sum 以兼容接口
        all_errors.append(dbit_error)
        count_error = 0
        qbit_keys = list(range(dbit + 1, rbit + 1))
        for qbit in qbit_keys:  # 方案2： 根据损失 的程度 确定补偿的比特位参数，并交替优化scale和bit位选择   # 从dbit+1 至 rbit 逐级递增
            max_q = torch.tensor(2 ** qbit - 1)
            # 预估新的 Scale (先假设为原来的一半，作为迭代的起点)
            # 注意：这里我们让 s_new 成为一个可优化的变量
            s_next = curr_scale / 2.0
            z_next = curr_zero * 2.0

            # === 核心改进：交替优化 (Coordinate Descent) ===
            # 迭代 2-3 次通常足以收敛。因为 Scale 和 LSB 是强耦合的。
            for _ in range(3):
                # --- 步骤 A: 修正 LSB 选择 (解决 Overshoot 问题) ---
                # 候选 1: 补 0 -> 整数值为 2*q_curr
                q0 = torch.clamp(q_curr * 2, 0, max_q).float() - z_next
                w_rec0 = s_next * q0
                # dist0 = (w - rec0).pow(2)  # 距离平方
                dist0 = torch.cosine_similarity(w, w_rec0, dim=1)  # 余弦相似度

                # 候选 2: 补 1 -> 整数值为 2*q_curr + 1
                q1 = torch.clamp(q_curr * 2 + 1, 0, max_q).float() - z_next
                w_rec1 = s_next * q1
                # dist1 = (w - rec1).pow(2)
                dist1 = torch.cosine_similarity(w, w_rec1, dim=1)  # 余弦相似度

                # 决策：谁的距离小选谁
                # pick_1 = (dist1 < dist0).float()    # # 距离平方    # 解决了 "残差为正但不需要进位" 的问题
                pick_1 = (dist1 > dist0).float().unsqueeze(1)

                # 生成新的整数 q_next
                q_next = torch.clamp(q_curr * 2 + pick_1, 0, max_q).long()


                # --- 步骤 B: 基于确定的整数，优化 Scale (MSE) ---
                v = q_next.float() - z_next
                # 带入 ir (重要性) 进行加权最小二乘
                # numerator: sum( ir * w * v )
                # denominator: sum( ir * v^2 )
                numerator = torch.sum(ir * w * v, dim=1, keepdim=True)
                denominator = torch.sum(ir * v * v, dim=1, keepdim=True)
                # numerator = torch.sum(w * v, dim=1, keepdim=True)
                # denominator = torch.sum(v * v, dim=1, keepdim=True)
                # 更新 Scale
                s_next = numerator / (denominator + 1e-12)

                # --- 步骤 C: 闭式解优化 Zero Point (z) ---
                # 这里的 z 不再是简单的整数翻倍，而是允许在连续空间寻找最优平移
                # Formula: z = [sum(H*Q) - sum(H*W)/s] / sum(H)
                sum_h = ir.sum(dim=1, keepdim=True) + 1e-12
                term1 = (ir * q_next).sum(dim=1, keepdim=True) / sum_h
                term2 = (ir * w).sum(dim=1, keepdim=True) / (s_next * sum_h)
                z_next = term1 - term2
                # z_next = torch.clamp(torch.round(z_next), 0, max_q)

            # === 循环结束，保存当前 bit 的最佳结果 ===

            # 计算最终误差 (使用 Weighted MSE 或 L1)
            w_rec_final = s_next * (q_next.float() - z_next)
            current_error = (w - w_rec_final).abs().sum()  # 依然返回 sum 以兼容接口


            all_errors.append(current_error)
            all_scales.append(s_next)
            all_zeros.append(z_next.squeeze())
            # 更新状态进入下一级
            curr_scale = s_next
            curr_zero = z_next
            q_curr = q_next
            # print('无效补偿触发次数',count_error)

        # --- 4. 结果保存与返回 ---
        self.qw_rbit = q_curr.to(torch.uint8)
        self.scales = torch.stack(all_scales, dim=0)  # 由[tensor(1), tensor(2)] 转为 tensor([1, 2])
        self.zeros = torch.stack(all_zeros, dim=0)
        return torch.stack(all_errors).unsqueeze(0)  # , torch.stack(error_noc).unsqueeze(0)
    
    
    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def quantize_int(self, x):
        if self.ready():
            return quantize_int(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        # 如果self.scale 是否全部元素都不为0，则返回True； 否则返回False
        return torch.all(self.scale != 0)


    def plot_hit(self, errorw, errorq, error3, e1, qbit, save_dir="comp_distribution_plots"):
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 将张量移动到CPU并转换为numpy数组
        error1_np = errorw.cpu().numpy().flatten()
        error2_np = errorq.cpu().numpy().flatten()
        error3_np = error3.cpu().numpy().flatten()
        e1_np = e1.cpu().numpy().flatten()[:10000]
        # e2_np = e2.cpu().numpy().flatten()

        # 计算统计量
        error1_mean = np.mean(error1_np)
        error1_std = np.std(error1_np)
        error1_sum = np.sum(error1_np)
        error2_mean = np.mean(error2_np)
        error2_std = np.std(error2_np)
        error2_sum = np.sum(error2_np)
        error3_mean = np.mean(error3_np)
        error3_std = np.std(error3_np)
        error3_sum = np.sum(error3_np)

        # 创建1行2列的子图
        plt.figure(figsize=(16, 10))  # 宽度增加以适应两个子图

        # 第一个子图：error1分布
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(error1_np, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=error1_mean, color='red', linestyle='--',
                    label=f'均值: {error1_mean:.3f}')
        ax1.axvline(x=error1_std, color='green', linestyle='--',
                    label=f'标准差: {error1_std:.3f}')
        ax1.set_xlabel('误差值')
        ax1.set_ylabel('频次')
        ax1.set_title(f"{self.name} - 权重W误差 (bit={qbit}), err_sum={error1_sum}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 第二个子图：error2分布
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(error2_np, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=error2_mean, color='red', linestyle='--',
                    label=f'均值: {error2_mean:.3f}')
        ax2.axvline(x=error2_std, color='green', linestyle='--',
                    label=f'标准差: {error2_std:.3f}')
        ax2.set_xlabel('误差值')
        ax2.set_ylabel('频次')
        ax2.set_title(f"{self.name} - 整数权重Q误差 (bit={qbit}), err_sum={error2_sum}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 第二个子图：error3分布
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(error3_np, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(x=error3_mean, color='red', linestyle='--',
                    label=f'均值: {error3_mean:.3f}')
        ax3.axvline(x=error2_std, color='green', linestyle='--',
                    label=f'标准差: {error3_std:.3f}')
        ax3.set_xlabel('误差值')
        ax3.set_ylabel('频次')
        ax3.set_title(f"{self.name} - 补偿后权重W误差 (bit={qbit}), err_sum={error3_sum}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 第二个子图：error3分布
        ax4 = plt.subplot(2, 1, 2)  # 第二行，占用整个宽度
        # 左侧y轴：
        color1 = 'tab:blue'
        ax4.set_xlabel("权重索引", fontsize=12)
        ax4.set_ylabel("Hessian对角线值", color=color1, fontsize=12)
        line1 = ax4.plot(range(len(e1_np)), e1_np,
                         color=color1, linewidth=1.5, label="q-8bit")
        ax4.tick_params(axis='y', labelcolor=color1)
        ax4.grid(True, alpha=0.3, axis='both')
        # # 创建右侧y轴：
        # ax5 = ax4.twinx()
        # color2 = 'tab:orange'
        # ax5.set_ylabel("激活值", color=color2, fontsize=12)
        # # 绘制激活值
        # line2 = ax5.plot(range(len(e2_np)), e2_np, color=color2,
        #                  linestyle='--', linewidth=1.5, alpha=0.8, label="q-comp")
        # ax5.tick_params(axis='y', labelcolor=color2)
        # # 添加图例
        # lines = line1 + line2
        # labels = [l.get_label() for l in lines]
        # ax4.legend(lines, labels, loc='upper right', fontsize=10)
        # 设置标题
        ax4.legend(loc='upper right', fontsize=10)
        ax4.set_title(f"Hessian对角线与激活值对比 ",
                      fontsize=14, fontweight='bold', pad=20)


        # 添加整体标题
        plt.suptitle(f"误差补偿机制对比 - {self.name} (count={self.count}, bit={qbit})",
                     fontsize=14, fontweight='bold')
        # 调整布局
        plt.tight_layout()  # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # 保存图片
        combined_save_path = os.path.join(save_dir, f"{self.name}_no{self.count}_bit{qbit}_comp_plot.png")
        plt.savefig(combined_save_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"分布对比图已保存至: {combined_save_path}")


__all__ = ["Quantizer"]
