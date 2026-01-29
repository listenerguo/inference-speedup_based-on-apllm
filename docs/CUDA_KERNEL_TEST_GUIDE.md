# CUDA Formula 反量化内核测试指南


---

## 目录

1. [概述](#1-概述)
2. [测试环境要求](#2-测试环境要求)
3. [测试前准备](#3-测试前准备)
4. [第一阶段：编译 CUDA 扩展](#4-第一阶段编译-cuda-扩展)
5. [第二阶段：反量化正确性测试](#5-第二阶段反量化正确性测试)
6. [第三阶段：APLinear 集成测试](#6-第三阶段aplinear-集成测试)
7. [第四阶段：端到端 PPL 评测](#7-第四阶段端到端-ppl-评测)
8. [常见问题与解决方案](#8-常见问题与解决方案)
9. [测试结果记录表](#9-测试结果记录表)

---

## 1. 概述

### 1.1 我们做了什么？

我们优化了大语言模型的量化推理速度。具体来说：

| 之前 (Python) | 之后 (CUDA) |
|---------------|-------------|
| 反量化在 Python 中执行，速度慢 | 反量化在 CUDA 内核中执行，速度快 |
| 需要生成完整的权重矩阵，占用大量显存 | 反量化与矩阵乘法融合，节省显存 |

### 1.2 新增的功能

- `dequant_formula_kbit`: CUDA 反量化函数
- `matmul_kbit_pergroup`: CUDA 融合矩阵乘法函数
- `APLinear` 自动使用 CUDA 加速

### 1.3 测试目标

1. ✅ 编译通过
2. ✅ 反量化结果与 Python 版本一致
3. ✅ APLinear 输出结果与 Python 版本一致
4. ✅ PPL 评测结果正确
5. ✅ 性能有明显提升

---

## 2. 测试环境要求

### 2.1 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU (支持 CUDA) |
| GPU 显存 | >= 8GB (推荐 16GB+) |
| 系统内存 | >= 16GB |

### 2.2 软件要求

| 软件 | 版本要求 | 检查命令 |
|------|----------|----------|
| CUDA | >= 11.0 | `nvcc --version` |
| Python | >= 3.8 | `python --version` |
| PyTorch | >= 2.0 | `python -c "import torch; print(torch.__version__)"` |
| GCC/G++ | >= 7.0 | `gcc --version` |

### 2.3 环境检查步骤

请依次执行以下命令，确认环境正确：

```bash
# 步骤 1: 检查 CUDA 版本
nvcc --version
```

**预期输出示例**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.1, V12.1.66
```

如果看到版本号，说明 CUDA 已安装。如果提示 "command not found"，请先安装 CUDA。

```bash
# 步骤 2: 检查 Python 版本
python --version
```

**预期输出示例**:
```
Python 3.10.12
```

如果提示 "command not found"，请尝试 `python3 --version`。

```bash
# 步骤 3: 检查 PyTorch 和 CUDA 是否可用
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**预期输出示例**:
```
PyTorch: 2.1.0
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

⚠️ **重要**: 如果 `CUDA available: False`，说明 PyTorch 没有正确安装 CUDA 版本。请重新安装：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. 测试前准备

### 3.1 进入项目目录

```bash
cd /path/to/inference-speedup_based-on-apllm
```

请将 `/path/to/` 替换为实际的项目路径。

### 3.2 检查文件是否存在

执行以下命令，确认新增的文件都在：

```bash
ls -la any_precision/modules/kernels/
```

**预期输出** (应该包含以下文件):
```
dequant.cuh
dequant_formula.cuh      <-- 新增
main.cu                  <-- 已修改
matmul.cuh               <-- 已修改
setup.py
```

```bash
ls -la test_*.py
```

**预期输出**:
```
test_aplinear_cuda.py    <-- 新增
test_dequant_formula.py  <-- 新增
```

### 3.3 设置 CUDA 环境变量

```bash
# 查找 CUDA 安装路径
which nvcc
```

**预期输出示例**:
```
/usr/local/cuda/bin/nvcc
```

根据输出设置环境变量：

```bash
# 设置 CUDA_HOME (根据上面的输出调整路径)
export CUDA_HOME=/usr/local/cuda

# 验证设置成功
echo $CUDA_HOME
```

**预期输出**:
```
/usr/local/cuda
```

---

## 4. 第一阶段：编译 CUDA 扩展

### 4.1 进入 kernels 目录

```bash
cd any_precision/modules/kernels
```

### 4.2 清理旧的编译文件 (可选)

```bash
rm -rf build/
rm -f *.so
```

### 4.3 执行编译

```bash
python setup.py install
```

### 4.4 检查编译结果

**成功的输出示例**:
```
running install
running build
running build_ext
building 'any_precision_ext' extension
...
creating build/lib.linux-x86_64-cpython-310
...
Installed /path/to/site-packages/any_precision_ext...
```

**关键检查点**:
- ✅ 没有红色的 `error` 字样
- ✅ 最后显示 `Installed ...`

### 4.5 验证安装成功

```bash
python -c "from any_precision_ext import dequant_formula_kbit, matmul_kbit_pergroup; print('Import SUCCESS!')"
```

**预期输出**:
```
Import SUCCESS!
```

### 4.6 编译失败怎么办？

如果编译失败，请记录完整的错误信息，并查看 [第8节 常见问题](#8-常见问题与解决方案)。

---

## 5. 第二阶段：反量化正确性测试

### 5.1 返回项目根目录

```bash
cd /path/to/inference-speedup_based-on-apllm
```

### 5.2 运行反量化测试

```bash
python test_dequant_formula.py
```

### 5.3 测试输出解读

**成功的输出示例**:
```
============================================================
  Formula Dequantization Verification Test
  w = scale * (w' - zero)
============================================================

Using device: NVIDIA GeForce RTX 3090

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
  Max absolute difference: 1.234567e-04
  Mean absolute difference: 5.678901e-06
  ✓ Results match within tolerance (rtol=0.001, atol=0.001)

... (更多测试配置) ...

============================================================
  ALL TESTS PASSED! ✓
============================================================
```

### 5.4 关键检查点

| 检查项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| CUDA extension loaded | ✓ | ___ |
| Shapes match | ✓ | ___ |
| Results match within tolerance | ✓ | ___ |
| ALL TESTS PASSED | ✓ | ___ |

### 5.5 性能数据记录

测试通过后，会显示性能对比：

```
[Python] 100 iterations...
  Average time: 12.345 ms

[CUDA] 100 iterations...
  Average time: 0.678 ms

[Summary]
  Python: 12.345 ms
  CUDA:   0.678 ms
  Speedup: 18.21x
```

请记录：
- Python 时间: ___ ms
- CUDA 时间: ___ ms
- 加速比: ___x

---

## 6. 第三阶段：APLinear 集成测试

### 6.1 运行集成测试

```bash
python test_aplinear_cuda.py
```

### 6.2 测试输出解读

**成功的输出示例**:
```
============================================================
  APLinear CUDA Integration Test
  Formula-based dequantization: w = s * (w' - z)
============================================================

Using device: NVIDIA GeForce RTX 3090

============================================================
Testing APLinear: in=1024, out=256
  group_size=128, w_bits=6
  batch_size=1, seq_len=32
============================================================
  CUDA Extension Available: True

[1] Running Python implementation...
  Output shape: torch.Size([1, 32, 256])
  Time: 15.234 ms

[2] Running CUDA implementation...
  Output shape: torch.Size([1, 32, 256])
  Time: 0.876 ms

[3] Comparing results...
  ✓ Shapes match: torch.Size([1, 32, 256])
  Max absolute difference: 2.345678e-03
  Mean absolute difference: 1.234567e-04
  ✓ Results match within tolerance (rtol=0.01, atol=0.01)

[4] Performance Summary
  Python: 15.234 ms
  CUDA:   0.876 ms
  Speedup: 17.39x

... (更多测试配置) ...

============================================================
  ALL TESTS PASSED! ✓
============================================================
```

### 6.3 关键检查点

| 检查项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| CUDA Extension Available | True | ___ |
| Output shapes match | ✓ | ___ |
| Results match within tolerance | ✓ | ___ |
| ALL TESTS PASSED | ✓ | ___ |

### 6.4 性能数据记录

对于 4096x4096 的矩阵：
- Python 时间: ___ ms
- CUDA 时间: ___ ms
- 加速比: ___x

---

## 7. 第四阶段：端到端 PPL 评测

### 7.1 准备工作

确认以下路径存在并正确配置：

```bash
# 检查量化模型目录
ls -la /path/to/ap-(opt-1.3b)-w8_orig4-gc128-c4_s100_blk256_0113_l2h/

# 检查数据集目录
ls -la /path/to/wikitext2/
```

### 7.2 修改配置 (如需要)

编辑 `run_ppl_exp-onetime.py`，确认以下配置正确：

```python
qbit_model_dir = "ap-(opt-1.3b)-w8_orig4-gc128-c4_s100_blk256_0113_l2h"
dataset_path = "wikitext2"
quant_bit = 6
```

### 7.3 运行 PPL 评测

```bash
python run_ppl_exp-onetime.py
```

### 7.4 预期输出

```
ap
ppl calculation of Any-precision_my:
Evaluating ...
0
1
2
...
23
14.567890  <-- PPL 值

ap-(opt-1.3b)-w8_orig4-gc128-c4_s100_blk256_0113_l2h量化后-6bit-模型  执行PPL计算所用时间：45.678
```

### 7.5 关键检查点

| 检查项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| 程序正常运行无报错 | ✓ | ___ |
| 输出 PPL 值 | 数值合理 (通常 10-50) | ___ |
| 执行时间 | 有明显改善 | ___ |

### 7.6 对比测试 (可选)

为了确认加速效果，可以临时禁用 CUDA 加速进行对比：

1. 编辑 `any_precision/modules/APLinear.py`
2. 将第一行的 `CUDA_AVAILABLE = True` 改为 `CUDA_AVAILABLE = False`
3. 重新运行 PPL 评测
4. 记录时间
5. 改回 `CUDA_AVAILABLE = True`

| 模式 | 执行时间 |
|------|----------|
| Python (禁用 CUDA) | ___ 秒 |
| CUDA (启用) | ___ 秒 |
| 加速比 | ___x |

---

## 8. 常见问题与解决方案

### 问题 1: `CUDA_HOME environment variable is not set`

**症状**:
```
OSError: CUDA_HOME environment variable is not set.
Please set it to your CUDA install root.
```

**解决方案**:
```bash
# 查找 CUDA 安装路径
which nvcc
# 输出例如: /usr/local/cuda-12.1/bin/nvcc

# 设置环境变量 (去掉 /bin/nvcc 部分)
export CUDA_HOME=/usr/local/cuda-12.1

# 重新编译
python setup.py install
```

---

### 问题 2: `ModuleNotFoundError: No module named 'any_precision_ext'`

**症状**:
```
ModuleNotFoundError: No module named 'any_precision_ext'
```

**解决方案**:
1. 确认编译成功 (查看编译输出是否有 `Installed ...`)
2. 如果编译成功但仍报错，尝试：
```bash
pip install -e any_precision/modules/kernels/
```

---

### 问题 3: 编译时报 `nvcc fatal: Unsupported gpu architecture`

**症状**:
```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**解决方案**:
需要根据你的 GPU 架构修改编译选项。

1. 查看你的 GPU 架构：
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
```
输出例如 `(8, 6)` 表示 compute_86

2. 编辑 `setup.py`，添加对应的架构支持

---

### 问题 4: 测试时 `Results do NOT match within tolerance`

**症状**:
```
✗ Results do NOT match within tolerance
Max absolute difference: 0.123456
```

**解决方案**:
1. 如果差异较小 (< 0.1)，可能是 FP16 精度问题，属于正常现象
2. 如果差异很大 (> 1.0)，请记录完整的测试输出并反馈

---

### 问题 5: `CUDA out of memory`

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小测试的矩阵大小
2. 在运行前清理 GPU 显存：
```bash
nvidia-smi
# 找到占用显存的进程 PID
kill -9 <PID>
```

---

### 问题 6: `ImportError: libcudart.so.XX: cannot open shared object file`

**症状**:
```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

**解决方案**:
```bash
# 添加 CUDA 库路径
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新运行测试
python test_dequant_formula.py
```

---

## 9. 测试结果记录表

请填写以下表格，并将结果反馈给开发人员：

### 9.1 环境信息

| 项目 | 信息 |
|------|------|
| 测试日期 | |
| 测试人员 | |
| 服务器名称 | |
| GPU 型号 | |
| CUDA 版本 | |
| PyTorch 版本 | |
| Python 版本 | |

### 9.2 编译结果

| 项目 | 结果 |
|------|------|
| 编译是否成功 | ✓ / ✗ |
| 如失败，错误信息 | |

### 9.3 反量化测试结果

| 项目 | 结果 |
|------|------|
| 测试是否通过 | ✓ / ✗ |
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





```

---

## 附录：快速测试命令汇总

```bash
# 0. 环境检查
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 1. 编译
cd any_precision/modules/kernels
export CUDA_HOME=/usr/local/cuda
python setup.py install

# 2. 反量化测试
cd ../../..
python test_dequant_formula.py

# 3. APLinear 测试
python test_aplinear_cuda.py

# 4. PPL 评测
python run_ppl_exp-onetime.py
```

---

**文档结束**

如有任何问题，请联系开发团队。
