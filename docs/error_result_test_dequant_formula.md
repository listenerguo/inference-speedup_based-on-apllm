
## 9. 测试结果记录表

请填写以下表格，并将结果反馈给开发人员：

### 9.1 环境信息

| 项目 | 信息 |
|------|------|
| 测试日期 | 2026.1.30 16:30|
| 测试人员 | Guo|
| 服务器名称 | personal_laptop	|
| GPU 型号 | NVIDIA GeForce RTX 4060 Laptop GPU|
| CUDA 版本 | cu12.4(Cuda compilation tools, release 12.6, V12.6.77)|
| PyTorch 版本 |2.6.0 |
| Python 版本 |Python 3.11.13|

### 9.2 编译结果

| 项目 | 结果 |
|------|----|
| 编译是否成功 | ✓  |
| 如失败，错误信息 |    |

### 9.3 反量化测试结果

| 项目 | 结果 |
|------|------|
| 测试是否通过 |  ✗ |
| Python 平均时间 | ___ ms |
| CUDA 平均时间 | ___ ms |
| 加速比 | ___x |

### 9.4 APLinear 集成测试结果

| 项目 | 结果 |
|------|------|
| 测试是否通过 | ✓ / ✗ |
| Python 平均时间 | ___ ms |
| CUDA 平均时间 | ___ ms |
| 加速比 | ___x |

### 9.5 PPL 评测结果

| 项目 | 结果 |
|------|------|
| 评测是否成功 | ✓ / ✗ |
| PPL 值 | |
| 执行总时间 | ___ 秒 |

### 9.6 其他问题或备注

```
请在此处记录测试过程中遇到的任何问题或异常情况：

出错：符合问题 4: ✗ Results do NOT match within tolerance； 且误差较大
具体执行结果：
============================================================
  Formula Dequantization Verification Test
  w = scale * (w' - zero)
============================================================

Using device: NVIDIA GeForce RTX 4060 Laptop GPU

============================================================
Testing: w_bits=4, group_size=128, N=256, K=1024
============================================================
✓ CUDA extension loaded successfully
  qweight shape: torch.Size([8, 256, 32])
  scale shape: torch.Size([256, 8])
  zero shape: torch.Size([256, 8])

[1] Running Python implementation...
  Result shape: torch.Size([256, 1024])

[2] Running CUDA implementation...
  Result shape: torch.Size([256, 1024])

[3] Comparing results...
  ✓ Shapes match: torch.Size([256, 1024])
  Max absolute difference: 8.669922e+00
  Mean absolute difference: 7.067784e-01
  ✗ Results do NOT match within tolerance

  Worst case at [155, 67]:
    Python: 5.175781
    CUDA:   -3.494141
    Diff:   8.669922e+00

============================================================
Testing: w_bits=6, group_size=128, N=256, K=2048
============================================================
✓ CUDA extension loaded successfully
  qweight shape: torch.Size([8, 256, 64])
  scale shape: torch.Size([256, 16])
  zero shape: torch.Size([256, 16])

[1] Running Python implementation...
  Result shape: torch.Size([256, 2048])

[2] Running CUDA implementation...
  Result shape: torch.Size([256, 2048])

[3] Comparing results...
  ✓ Shapes match: torch.Size([256, 2048])
  Max absolute difference: 3.953125e+01
  Mean absolute difference: 3.044964e+00
  ✗ Results do NOT match within tolerance

  Worst case at [240, 940]:
    Python: 23.937500
    CUDA:   -15.593750
    Diff:   3.953125e+01

============================================================
Testing: w_bits=8, group_size=128, N=512, K=2048
============================================================
✓ CUDA extension loaded successfully
  qweight shape: torch.Size([8, 512, 64])
  scale shape: torch.Size([512, 16])
  zero shape: torch.Size([512, 16])

[1] Running Python implementation...
  Result shape: torch.Size([512, 2048])

[2] Running CUDA implementation...
  Result shape: torch.Size([512, 2048])

[3] Comparing results...
  ✓ Shapes match: torch.Size([512, 2048])
  Max absolute difference: 1.625625e+02
  Mean absolute difference: 1.211609e+01
  ✗ Results do NOT match within tolerance

  Worst case at [223, 439]:
    Python: 68.750000
    CUDA:   -93.812500
    Diff:   1.625625e+02

============================================================
Testing: w_bits=6, group_size=64, N=256, K=1024
============================================================
✓ CUDA extension loaded successfully
  qweight shape: torch.Size([8, 256, 32])
  scale shape: torch.Size([256, 16])
  zero shape: torch.Size([256, 16])

[1] Running Python implementation...
  Result shape: torch.Size([256, 1024])

[2] Running CUDA implementation...
  Result shape: torch.Size([256, 1024])

[3] Comparing results...
  ✓ Shapes match: torch.Size([256, 1024])
  Max absolute difference: 3.367188e+01
  Mean absolute difference: 3.284557e+00
  ✗ Results do NOT match within tolerance

  Worst case at [166, 297]:
    Python: 16.031250
    CUDA:   -17.640625
    Diff:   3.367188e+01

============================================================
Testing: w_bits=6, group_size=256, N=256, K=2048
============================================================
✓ CUDA extension loaded successfully
  qweight shape: torch.Size([8, 256, 64])
  scale shape: torch.Size([256, 8])
  zero shape: torch.Size([256, 8])

[1] Running Python implementation...
  Result shape: torch.Size([256, 2048])

[2] Running CUDA implementation...
  Result shape: torch.Size([256, 2048])

[3] Comparing results...
  ✓ Shapes match: torch.Size([256, 2048])
  Max absolute difference: 3.333594e+01
  Mean absolute difference: 2.573310e+00
  ✗ Results do NOT match within tolerance

  Worst case at [105, 1443]:
    Python: -18.296875
    CUDA:   15.039062
    Diff:   3.333594e+01

============================================================
  SOME TESTS FAILED! ✗
============================================================

进程已结束，退出代码为 1


#代码 测试：
原始python实现时，代码如下：
            # Fallback to Python implementation
            weight = self._dequant_temp(w_bits, self.group_size, self.qweight, scale, zero)
            out = torch.matmul(x, weight.T)
CUDA设计后，代码如下：
            out = matmul_kbit_pergroup(
                x_half,
                self.qweight,
                scale_half,
                zero_half,
                w_bits,
                self.group_size
            )
如何确定 上述 matmul_kbit_pergroup过程中的 两个步骤：
(1)完成反量化 得到 weight，(2)进而实现推理计算 torch.matmul(x, weight.T) ， 
导致  "✗ Results do NOT match within tolerance" 产生的原因 ：
是解包出错， 还是反量化与分组参数scale和zero 的计算出错， 还是最后于输入x的推理计算出错。

```

---