# run_new_pipeline.py — 基于预处理数据的全自动分层流水线

## 概述

`run_new_pipeline.py` 是简化后的新版本分层流水线。与 `run_auto_pipeline.py` 不同，它**不需要 4x 图像、扫描元数据或坐标映射**。它直接读取预处理后的输入，在 DAPI 图像上执行细胞分割，并生成层分析结果，不包含多余的可视化及中间产物。

---

## 输入格式

每个样本的输入文件位于 `dataset/inputs/preprocessed/<sample>/` 下：

| 文件 | 说明 |
|---|---|
| `dapi.png` / `dapi.tif` | DAPI / 40x 图像，用于细胞分割 |
| `pia.csv` | 灰质外边界（x, y 坐标列），等价于原 GM.csv |
| `white.csv` | 白质边界（x, y 坐标列），等价于原 WM.csv |
| `graymask.png` | 灰质二值掩膜（0 / 255），与 DAPI 图像同尺寸 |

## 输出格式

结果保存在 `dataset/outputs/<sample>/` 下：

| 文件 | 说明 |
|---|---|
| `cell_centroid.csv` | 细胞点云坐标（cell_id, centroid_x, centroid_y, area_px, area_um2） |
| `segmented_depth.csv` | 层深度边界（更名自 segmented_layers.csv） |
| `pia_boundary.csv` | pia 边界点（从输入复制） |
| `white_boundary.csv` | white 边界点（从输入复制） |
| `boundary_L1_2.csv` | L1/L2 轮廓坐标（x, y） |
| `boundary_L3_4.csv` | L3/L4 轮廓坐标（x, y） |
| `boundary_L4_5.csv` | L4/L5 轮廓坐标（x, y） |
| `layers_color_mask.png` | 多色逐像素层掩膜 |
| `layer_lines.png` | 所有边界线（pia + white + 分层线），以平滑白色粗虚线绘制 |
| `depth_density_layers_peak_based.png` | peak-based 分层算法诊断图 |

## 相对于 run_auto_pipeline.py 的变更

### 已移除
- **4x 图像处理** — 无组织分割、无坐标映射、无仿射变换
- **扫描 JSON 元数据** — 无需 4x.json / 40x.json
- **细胞分割可视化** — 无 `*_mask_color.png`、`*_overlay.png`、`*_full_mask.tif`（仅保留内部压缩标签掩膜）
- **多余的中间产物** — 无 `OuterInnerPoints_*.png`、`tissueMask.png`、`whiteMask.png`、`grayMask.png`
- **多余的最终可视化** — 无 `layers_overlay.png`、`combined_visualization.png`（运行后自动清理）

### 新增
- **`pia_boundary.csv` / `white_boundary.csv`** — 输入边界的直接拷贝
- **`boundary_L1_2.csv` / `boundary_L3_4.csv` / `boundary_L4_5.csv`** — 从深度场提取的分层线轮廓坐标
- **全白虚线 `layer_lines.png`** — pia、white 和所有内部分层线统一用白色粗虚线绘制（而非原版用绿色/红色区分）

### 保持不变的核心算法
- 基于 Cellpose 的分块细胞分割（使用 graymask 跳过无效区域）
- 基于欧几里得距离变换的逐像素深度场
- KDE 密度估计 + peak-based 自动分层

## 关键设计决策

1. **层边界映射**：根据 `segmented_depth.csv` 中的层名称（"1"、"2/3" 或 "3"、"4"）映射到请求的边界名称 `L1_2` / `L3_4` / `L4_5`。无论是否合并 L2/L3（`--separate-l23`），该映射均能正确工作。

2. **All-white layer_lines.png**：与使用绿色（GM/pia）和红色（WM）的旧版可视化不同，此流水线自行渲染 `layer_lines.png`，所有边界线均为白色粗虚线。

3. **graymask 双重用途**：在 Cellpose 瓦片分割期间跳过无效瓦片，以及在密度分析前过滤细胞。

4. **细胞分割可跳过**：使用 `--skip-cellpose` 复用已有的 `cell_centroids.csv`，仅重新运行分层步骤。

5. **样本级输出**：每个样本独立处理并输出至独立子目录。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-root` | `dataset/inputs/preprocessed` | 样本输入根目录 |
| `--output-root` | `dataset/outputs` | 样本输出根目录 |
| `--samples` | 全部子目录 | 指定处理的样本名，如 `--samples 1 5 10` |
| `--no-gpu` | 使用 GPU | 在 CPU 上运行 Cellpose |
| `--batch-size` | 64 | Cellpose 内部批大小 |
| `--tile-size` | 4096 | Cellpose 瓦片大小 |
| `--overlap` | 64 | Cellpose 瓦片重叠像素 |
| `--no-tta` / `--tta` | 禁用 TTA | Cellpose 测试时增强 |
| `--downsample-rate` | 0.8 | Cellpose 推理降采样率 |
| `--cellpose-python` | 自动检测 | 指定含 Cellpose 的 Python 可执行文件 |
| `--depth-method` | `legacy` | 深度计算方法（`legacy` / `harmonic`） |
| `--separate-l23` | 合并 L2/3 | 不合并 L2/L3，输出 5 层 |
| `--compact` | — | 等同于 `--compact-rate 0.1` |
| `--compact-rate` | 0.1 | 可视化降采样比例 |
| `--skip-cellpose` | — | 跳过细胞分割，使用已有 cell_centroids.csv |

## 使用示例

```bash
# 处理所有样本
python run_new_pipeline.py

# 处理特定样本
python run_new_pipeline.py --samples 1 5 10

# 指定自定义根目录
python run_new_pipeline.py --input-root dataset/inputs/preprocessed --output-root dataset/outputs

# 不合并 L2/3（输出 5 层）
python run_new_pipeline.py --separate-l23

# 仅重新运行分层（跳过 Cellpose）
python run_new_pipeline.py --skip-cellpose

# CPU 模式 + 大瓦片
python run_new_pipeline.py --no-gpu --tile-size 2048

# 完整自定义运行
python run_new_pipeline.py --samples 1 2 --no-gpu --depth-method harmonic --tile-size 3072
```
