# AutoLaminarSegment — 自动皮层分层分割与精炼

## 流程概览

```
preprocessed data (dapi.png, pia.csv, white.csv, graymask.png)
        │
        ▼
┌─────────────────────┐
│  Step 1: 粗分层      │  run_new_pipeline.py
│  peak-based 算法     │  → all_boundaries.csv
│                     │  → layers_color_mask.png
│                     │  → segmented_depth.csv
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Step 2: NN 精炼     │  src/refine_deform.py
│  ResMLP 液化微调     │  → all_boundaries_deformed.csv
│  训练 1-10, 推全样本  │  → layers_color_mask_refined.png
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Step 3: 评估分析     │  run_analysis.py
│                     │  → analysis_results.csv
│                     │  → summary/ 统计与可视化
└─────────────────────┘
```

---

## Step 1: 粗分层

```powershell
.\venv-cellpose\Scripts\python run_new_pipeline.py
```

输入（每个样本 `dataset/inputs/preprocessed/<n>/`）：
| 文件 | 说明 |
|------|------|
| `dapi.png` | DAPI 染色图像 (40x) |
| `pia.csv` | 灰质外边界 (GM, 即 pia 表面) |
| `white.csv` | 白质边界 (WM) |
| `graymask.png` | 灰质二值掩膜 |

输出（`dataset/outputs/<n>/`）：
| 文件 | 说明 |
|------|------|
| `cell_centroids.csv` | Cellpose 分割的细胞中心点 |
| `segmented_depth.csv` | peak-based 分层深度阈值 |
| **`all_boundaries.csv`** | **5 条全分辨率边界线**（pia, L1_2, L3_4, L4_5, white） |
| `layers_color_mask.png` | 粗分层彩色图 |
| `layer_lines.png` | 边界线可视化 |

关键参数：
- `--samples 1 2 3` — 只处理指定样本
- `--no-gpu` — CPU 模式
- `--skip-cellpose` — 跳过细胞分割，复用已有 centroids

---

## Step 2: NN 边界精炼（液化微调）

```powershell
# 完整训练 + 推理
.\venv-cellpose\Scripts\python src/refine_deform.py --blend 0.4
```

用 ResMLP 网络学习从局部细胞密度模式到 GT 边界偏移的映射。

### 算法

1. **特征**：沿边界法线方向采样 **31 点密度剖面**（半径 250px, grid_step=50）
2. **模型**：ResMLP — 输入 36 维 → 128 → 64 → 32 → 1（Δy 偏移）
3. **训练**：样本 1-10 作为训练集，样本 11-12 作为测试集
4. **平滑**：PCHIP 保形插值 + 高斯后滤波 + 局部异常修正 + 边缘渐变约束

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--blend` | 0.4 | 形变混合比例。1.0=全量精炼，0.4=适度，0.2=微调 |
| `--model` | resmlp | 模型架构（resmlp / mlp） |
| `--train` | 1-10 | 指定训练样本 |
| `--predict` | all | 指定推理样本 |

输出：
| 文件 | 说明 |
|------|------|
| **`all_boundaries_deformed.csv`** | **NN 精炼后的边界线**（与 all_boundaries.csv 格式一致） |
| `layers_color_mask_refined.png` | 精炼分层图（与粗分层对比） |
| `comparison_refined.png` | 三栏对比：Coarse vs Refined vs GT |
| `models/refine_deform/` | 训练好的模型权重 |

### 效果

| 边界 | Coarse MAE | Refined MAE (blend=0.4) |
|------|-----------|-------------------------|
| L1_2 | 35.5 px | **25.7 px (+28%)** |
| L3_4 | 78.6 px | **57.9 px (+26%)** |
| L4_5 | 51.7 px | **39.2 px (+24%)** |

---

## Step 3: 结果评估

```powershell
.\venv-cellpose\Scripts\python run_analysis.py
```

计算每个样本的 IoU / Dice / Precision / Recall / 边界误差 / 厚度误差，
输出到 `dataset/analysis/<n>/analysis_results.csv`。

汇总统计和可视化图表在 `dataset/analysis/summary/`。

---

## 数据目录结构

```
dataset/
├── inputs/
│   ├── preprocessed/<n>/    # 预处理输入
│   │   ├── dapi.png         # DAPI 图像
│   │   ├── pia.csv          # pia 边界
│   │   ├── white.csv        # white 边界
│   │   └── graymask.png     # 灰质掩膜
│   └── label/<n>/           # 人工标注 GT
│       ├── label_mask.png
│       └── label_boundaries.csv
├── outputs/<n>/             # 粗分层 + 精炼结果
│   ├── cell_centroids.csv
│   ├── segmented_depth.csv
│   ├── all_boundaries.csv
│   ├── all_boundaries_deformed.csv    ← NN 精炼
│   ├── layers_color_mask.png
│   └── layers_color_mask_refined.png  ← NN 精炼
└── analysis/                # 分析结果
    └── summary/
        └── stats_summary.csv
```

## 环境

使用 `venv-cellpose` 虚拟环境：
```powershell
.\venv-cellpose\Scripts\python <script.py>
```

依赖：PyTorch 2.4, OpenCV, scipy, scikit-learn, cellpose, matplotlib, pandas
