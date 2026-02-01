# GPU 加速 Bootstrap 分析 - 使用指南

## 📋 概述

为了解决 B=1000 时运行速度慢的问题，提供了两个优化版本：

1. **`fan_vote_certainty_analysis_fast.py`** ⭐ **推荐使用**
   - 使用多进程并行加速
   - 无需 GPU，在任何电脑上都能运行
   - 速度提升：8核 CPU 约 8 倍加速

2. **`fan_vote_certainty_analysis_gpu.py`**
   - 使用 GPU (CUDA) 加速矩阵运算
   - 需要 NVIDIA GPU 和 CUDA 环境
   - 适合有 GPU 的用户

---

## 🚀 快速开始（推荐方案）

### 使用多进程版本（无需 GPU）

```bash
# 直接运行，使用 1000 个 bootstrap 样本
python fan_vote_certainty_analysis_fast.py --B 1000

# 指定进程数（根据 CPU 核心数调整）
python fan_vote_certainty_analysis_fast.py --B 1000 --processes 8

# 只运行方法二（Bootstrap）
python fan_vote_certainty_analysis_fast.py --B 1000 --no-method1
```

**性能对比：**
- 原版单进程：B=15 约需 2-3 分钟，B=1000 约需 **2-3 小时**
- 多进程版本（8核）：B=1000 约需 **15-30 分钟**（速度提升约 6-8 倍）

---

## 💻 GPU 版本安装（可选）

如果你有 NVIDIA GPU 并想使用 GPU 加速：

### 1. 检查 CUDA 版本

```bash
nvidia-smi
```

查看 CUDA Version（如：12.1）

### 2. 安装 CuPy

根据你的 CUDA 版本选择：

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 不确定版本时，使用自动检测
pip install cupy
```

### 3. 运行 GPU 版本

```bash
python fan_vote_certainty_analysis_gpu.py --B 1000
```

**注意：** GPU 版本在当前实现中可能不如多进程版本快，因为瓶颈在于优化求解器而非矩阵运算。

---

## 📊 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--B` | 1000 | Bootstrap 样本数量 |
| `--sigma` | 0.01 | 噪声标准差 |
| `--processes` | CPU核心数-1 | 并行进程数（仅多进程版本） |
| `--csv` | 自动检测 | 输入 CSV 文件路径 |
| `--excel` | 自动检测 | 原始 Excel 文件路径 |
| `--out-dir` | 当前目录 | 输出目录 |
| `--no-method1` | False | 跳过方法一 |
| `--no-method2` | False | 跳过方法二 |

### 建议配置

**根据你的计算机配置选择：**

- **4 核 CPU**：`--processes 3 --B 1000`（约 40-60 分钟）
- **8 核 CPU**：`--processes 7 --B 1000`（约 15-30 分钟）
- **16 核 CPU**：`--processes 15 --B 1000`（约 8-15 分钟）

---

## 📈 输出文件

运行完成后会生成以下文件：

1. **`certainty_method1_interval_summary.csv`**
   - 方法一（可行域区间法）的汇总统计

2. **`certainty_method2_bootstrap_1000.csv`**
   - 方法二（Bootstrap）的完整结果
   - 包含每个选手的：均值、方差、标准差、95% 置信区间

3. **`certainty_combined_1000.csv`**
   - 合并了两种方法的结果

---

## 🔍 性能优化原理

### 多进程版本（推荐）

```
原版：串行处理
[样本1] → [样本2] → [样本3] → ... → [样本1000]  ⏱️ 很慢

多进程版本：并行处理
[样本1]   [样本9]   [样本17]  ...
[样本2]   [样本10]  [样本18]  ...  ⏱️ 快 8 倍
[样本3]   [样本11]  [样本19]  ...
...       ...       ...
```

**优点：**
- ✅ 无需额外安装库
- ✅ 充分利用多核 CPU
- ✅ 线性加速（8核约8倍速度）
- ✅ 内存占用合理

### GPU 版本

将噪声生成等矩阵运算移到 GPU，但由于优化求解器仍在 CPU，整体加速效果有限。

---

## 🐛 常见问题

### Q1: 提示 "No module named 'fan_vote_estimation_entropy_smooth'"

**解决：** 确保在 `task1-1` 目录下运行，且该目录下有 `fan_vote_estimation_entropy_smooth.py` 文件。

```bash
cd d:\Users\13016\Desktop\26MCM\2026_C\task1-1
python fan_vote_certainty_analysis_fast.py --B 1000
```

### Q2: 多进程版本报错 "can't pickle ..."

**解决：** 在 Windows 上，确保在 `if __name__ == '__main__':` 块内运行代码。
脚本已经正确处理，直接运行即可。

### Q3: 内存不足

**解决：** 减少进程数

```bash
python fan_vote_certainty_analysis_fast.py --B 1000 --processes 4
```

### Q4: 想看更详细的进度

脚本已包含实时进度显示，会显示：
- 完成百分比
- 已用时间
- 预计剩余时间

---

## 📝 使用示例

### 示例 1：标准运行

```bash
python fan_vote_certainty_analysis_fast.py --B 1000
```

输出：
```
======================================================================
观众投票确定性分析 - 高性能版本
======================================================================
输出目录: d:\Users\13016\Desktop\26MCM\2026_C\task1-1

----------------------------------------------------------------------
方法二：多进程并行 Bootstrap 分析
----------------------------------------------------------------------
  Bootstrap 样本数: 1000
  并行进程数: 8
  噪声标准差: 0.01
  平滑参数: 100.0

开始处理...
  完成: 10/1000 (1.0%) | 已用时: 15.3s | 预计剩余: 1515.0s
  完成: 20/1000 (2.0%) | 已用时: 30.1s | 预计剩余: 1475.0s
  ...
  完成: 1000/1000 (100.0%) | 已用时: 1234.5s | 预计剩余: 0.0s

✓ Bootstrap 完成！总用时: 1234.5s (20.58 分钟)
```

### 示例 2：快速测试（小样本）

```bash
# 先用 B=50 测试
python fan_vote_certainty_analysis_fast.py --B 50 --processes 4
```

### 示例 3：只运行 Bootstrap

```bash
python fan_vote_certainty_analysis_fast.py --B 1000 --no-method1
```

---

## ⚡ 性能对比总结

| 版本 | B=15 | B=100 | B=1000 | 需要 GPU |
|------|------|-------|--------|----------|
| 原版 | 2-3 分钟 | 15-20 分钟 | 2-3 小时 | ❌ |
| 多进程（8核） | 20-30 秒 | 2-3 分钟 | **15-30 分钟** | ❌ |
| GPU 版本 | 类似多进程 | 类似多进程 | 类似多进程 | ✅ |

**结论：推荐使用多进程版本（`fan_vote_certainty_analysis_fast.py`）** 🎯

---

## 📞 技术支持

如有问题，请检查：
1. Python 版本 ≥ 3.8
2. 所需文件都在正确位置
3. 有足够的磁盘空间（约 100MB）
4. 内存充足（建议 ≥ 8GB）

---

**祝运行顺利！🚀**
