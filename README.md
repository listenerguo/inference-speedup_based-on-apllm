# inference-speedup_based-on-apllm
inference speedup 


---

# 任意精度 LLM 量化推理与 PPL 评测框架

**Any-Precision LLM Inference Framework**

本项目是一个面向大语言模型（LLM）的 **任意精度量化推理的评测框架**，支持两套**互不兼容的反量化推理体系**，并以统一入口完成模型推理、PPL 计算与性能统计。
互不兼容的反量化推理体系:
（1）基于w = LUT(w') 的查找表反量化推理(LUT方案)：https://github.com/SNU-ARC/any-precision-llm
（2）基于w = s(w'-z) 的计算反量化推理(Ours方案)

框架的核心目标是：

> **将 LUT方案 修改为 Ours方案，并将 以实现的 Python 端的Ours方案反量化逻辑 优化为 CUDA 内核，实现“位平面解包 + 反量化”的融合推理加速。**

---

## 1. 统一执行链路（不分量化体系）

### 统一入口脚本

**`run_ppl_exp-onetime.py`**

两套量化体系均通过该脚本完成完整实验流程。

### 执行流程

1. **加载量化模型与数据集**

   * 模型类型（二选一）：

     * `APForCausalLM`
     * `AnyPrecisionForCausalLM`
   * 数据集：`WikiText2`

2. **调用 `opt_eval`（推理 + PPL 计算）**

   * 数据预处理：序列切分、输入缓存
   * 逐层调用 Decoder Layer 的 `forward`
   * 每层进入量化 Linear 的 `forward`（反量化 + 矩阵乘法）

3. **计算 Loss 与 PPL**
   
   * 基于 logits 计算 `CrossEntropyLoss`
   * 得到 PPL（Perplexity）

4. **输出结果**

   * PPL 数值
   * 端到端推理耗时（从加载到 PPL 完成）

> **结论**：推理、PPL 与性能统计被封装为一体化流程，中间层无需修改即可切换量化体系。

---

## 2. 关键文件与职责

| 路径                                                 | 作用                                                           |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `run_ppl_exp-onetime.py`                           | **统一入口**：加载模型/数据集、触发评测、统计耗时                   |
| `any_precision/evaluate/eval.py`                   | **评测辅助模块**：输入缓存、序列切分等                             |
| `any_precision/modules/APForCausalLM.py`           | **AP 体系模型封装**：动态精度切换，调用 `APLinear`                       (Ours方案) |
| `any_precision/modules/APLinear.py`                | **AP 体系核心 Linear**：解包 + 反量化 + 推理（当前主要为 Python 实现）    (Ours方案) |
| `any_precision/modules/AnyPrecisionForCausalLM.py` | **LUT 体系模型封装**：调用 `AnyPrecisionLinear`                         (LUT方案)  |
| `any_precision/modules/AnyPrecisionLinear.py`      | **LUT 体系核心 Linear**：解包 + LUT 反量化 + 推理（CUDA）                (LUT方案)  |
| `any_precision/modules/kernels/`                   | **CUDA 核心算子目录**：接口 / 解包 / 融合 GEMM                           (LUT方案)  |

---

## 3. 两套量化反量化体系（互不兼容）

### A. LUT 查表体系（原仓库方案）

* **量化模型来源**
  SNU-ARC [`any-precision-llm`](https://github.com/SNU-ARC/any-precision-llm)
  通过 `quantize.py` 生成

* **模型类型**
  `AnyPrecisionForCausalLM`

* **反量化逻辑**

  1. `w_{bp} → w'`（位平面解包）
  2. `w = LUT(w')`（查表映射为浮点权重）

* **执行栈**

  * Python：`AnyPrecisionForCausalLM.py` / `AnyPrecisionLinear.py`
  * CUDA：`kernels/main.cu` / `dequant.cuh` / `matmul.cuh`
  * 触发点：`AnyPrecisionLinear.forward()`

> 该体系已完整 CUDA 化，但仅适用于原仓库量化模型。
---

### B. Ours方案（Scale-Zero）体系（目标方案）

* **量化模型来源**
  `my-quantization` 自研量化流程
  (上传量化后模型--网盘链接--todo)
  
* **模型类型**
  `APForCausalLM`

* **反量化公式**

  ```text
  w = s · (w' − z)
  ```

  * `w'`：位平面解包后的整数权重
  * `s`：缩放因子（Scale）
  * `z`：零点（Zero Point）

* **当前实现**

  * Python：`APForCausalLM.py` + `APLinear.py`
  * 反量化通过 `APLinear.forward()` + `_dequant_temp` 实现

* **目标实现**

  * 用 CUDA 内核（MY_kernels）替换 `_dequant_temp`
  * 复用：

    * `main.cu`（接口）
    * `dequant.cuh`（解包 + 公式反量化）
    * `matmul.cuh`（融合 GEMM） %后续优化考虑

> 该体系模型已就绪、推理可跑，但反量化在 Python，是主要性能瓶颈。

---

## 4. CUDA Kernels 目录说明与改造目标

### 现有 Kernels（LUT 体系）

* `main.cu`
  CUDA 对外接口（Python Extension）

* `dequant.cuh`

  * 位平面解包：`w_{bp} → w'`
  * LUT 查表反量化：`w = LUT(w')`

* `matmul.cuh`

  * 反量化 + 矩阵乘法融合

---

### 改造目标（Formula 体系）

* **保留**

  * 位平面解包逻辑：`w_{bp} → w'`

* **替换**

  * 从：`w = LUT(w')`
  * 到：`w = s · (w' − z)`

* **最终目标**

  * 在 CUDA 内核中完成：

    * 解包
    * 公式反量化
    * GEMM 融合
  * 完全替代 `APLinear.py` 中的 Python `_dequant_temp`

---

## 5. 运行方式

统一执行：

```bash
python run_ppl_exp-onetime.py
```

只需在 `main()` 中配置以下参数：

```python
qbit_model_dir = "path/to/quantized_model"  # 量化模型路径（需与体系匹配）
dataset_path  = "path/to/WikiText2"
quant_bit     = 6  # 推理位宽（w_bits）
```

---

## 6. Linear 层关键参数对比

### APLinear（Formula-based）

* `self.qweight`
  8-bit 整数权重（位平面格式）

* `self.group_size`
  每组 Scale / Zero 覆盖的列数

* `self._buffers[f'scale{w_bits}']`
  缩放因子 `s`

* `self._buffers['zero']`
  零点 `z`

* `w_bits`
  实际推理使用的位宽（3 / 4 / 6 / 8）

---

### AnyPrecisionLinear（LUT-based）

* `self.qweight`
  量化权重

* `self._buffers[f'lut{w_bits}']`
  LUT 查找表（整数 → 浮点）

* `w_bits`
  查表使用的有效位宽

---

## 7. 总结

* 当前维护两条可运行的推理链路：

  * **LUT 体系**

    * 适用于非均匀量化
    * 已完全 CUDA 化
    * 依赖原仓库量化模型
  * **Formula 体系**

    * 适用于均匀量化
    * 数值正确、模型灵活
    * 反量化仍在 Python，性能待优化

* **核心改造方向**

  > 在保持位平面解包不变的前提下，将反量化从 LUT 查表切换为
  > `w = s · (w' − z)`，并与 GEMM 深度融合至 CUDA 内核。

---
