"""
分层结果验证分析代码
比较算法输出的 layers_color_mask.png 与 groundtruth.png

颜色映射（BGR格式）：
Ground Truth:
  - 0x1c19fc (25, 28, 252) → L1
  - 0x7e7e12 (18, 126, 126) → L2/3
  - 0x4cffff (255, 255, 76) → L4
  - 0xf87f7f (127, 127, 248) → L5/6

Algorithm Output:
  - 0xf86464 (100, 100, 248) → L1
  - 0x76fe6d (109, 254, 118) → L2/3
  - 0x6768fc (252, 104, 103) → L4
  - 0xfefe6d (109, 254, 254) → L5/6
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)


def resolve_image_path(directory: Path, filename: str) -> Path:
    base = directory / filename
    if base.exists():
        return base

    stem = base.stem if base.suffix else base.name
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = base.parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return base


class LayerResultAnalyzer:
    """分层结果分析器"""
    
    # Ground Truth颜色映射（BGR格式）
    GT_LAYER_COLORS = {
        'L1': (255, 100, 100),
        'L2/3': (100, 255, 100),
        'L4': (100, 100, 255),
        'L5/6': (255, 100, 255),
    }
    
    # 算法输出颜色映射（BGR格式）
    ALGO_LAYER_COLORS = {
        'L1': (255, 100, 100),
        'L2/3': (100, 255, 100),
        'L4': (100, 100, 255),
        'L5/6': (255, 255, 100),
    }
    
    def __init__(self, output_dir="output", input_dir="input", groundtruth_path=None):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.groundtruth_path = Path(groundtruth_path) if groundtruth_path else None
        
        # 加载图像
        self.algo_mask = None
        self.gt_mask = None
        
        # 解析结果
        self.algo_layers = {}  # layer_name -> binary mask
        self.gt_layers = {}    # layer_name -> binary mask
        
        # 标准层名
        self.standard_layers = ['L1', 'L2/3', 'L4', 'L5/6']
    
    def load_images(self):
        """加载算法输出和ground truth图像"""
        algo_path = resolve_image_path(self.output_dir, "layers_color_mask.png")
        if self.groundtruth_path is not None:
            gt_path = self.groundtruth_path
        else:
            gt_path = resolve_image_path(self.input_dir, "groundtruth.png")

        
        if not algo_path.exists():
            raise FileNotFoundError(f"算法输出文件不存在: {algo_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground Truth文件不存在: {gt_path}")
        
        self.algo_mask = cv2.imread(str(algo_path))
        self.gt_mask = cv2.imread(str(gt_path))
        
        print(f"算法输出尺寸: {self.algo_mask.shape}")
        print(f"Ground Truth尺寸: {self.gt_mask.shape}")
        
        # 检查尺寸是否一致
        if self.algo_mask.shape != self.gt_mask.shape:
            print("警告: 图像尺寸不一致，将调整Ground Truth尺寸")
            self.gt_mask = cv2.resize(self.gt_mask, 
                                       (self.algo_mask.shape[1], self.algo_mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        
        return self
    
    def parse_masks_by_color(self):
        """根据固定颜色映射解析两张mask图像"""
        print("\n" + "="*50)
        print("解析图像（基于固定颜色映射）...")
        print("="*50)
        
        tolerance = 15  # 颜色容差
        
        # 解析算法输出
        print("\n算法输出:")
        for layer_name, color_bgr in self.ALGO_LAYER_COLORS.items():
            mask = self._extract_color_mask(self.algo_mask, color_bgr, tolerance)
            pixel_count = np.sum(mask > 0)
            self.algo_layers[layer_name] = mask
            print(f"  {layer_name}: BGR{color_bgr} → {pixel_count} pixels")
        
        # 解析Ground Truth
        print("\nGround Truth:")
        for layer_name, color_bgr in self.GT_LAYER_COLORS.items():
            mask = self._extract_color_mask(self.gt_mask, color_bgr, tolerance)
            pixel_count = np.sum(mask > 0)
            self.gt_layers[layer_name] = mask
            print(f"  {layer_name}: BGR{color_bgr} → {pixel_count} pixels")
        
        return self
    
    def _extract_color_mask(self, image, target_color, tolerance=15):
        """从图像中提取指定颜色的mask"""
        target = np.array(target_color, dtype=np.uint8)
        
        # 计算颜色距离
        diff = np.abs(image.astype(np.int16) - target)
        mask = np.all(diff <= tolerance, axis=2).astype(np.uint8) * 255
        
        return mask
    
    def calculate_iou(self, mask1, mask2):
        """计算两个mask的IoU（交并比）"""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def calculate_dice(self, mask1, mask2):
        """计算Dice系数"""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        sum_pixels = (mask1 > 0).sum() + (mask2 > 0).sum()
        
        if sum_pixels == 0:
            return 0.0
        return 2 * intersection / sum_pixels
    
    def calculate_overlap_metrics(self):
        """计算每层的重叠率指标"""
        print("\n" + "="*50)
        print("计算各层重叠率...")
        print("="*50)
        
        results = []
        
        for layer_name in self.standard_layers:
            algo_mask = self.algo_layers.get(layer_name)
            gt_mask = self.gt_layers.get(layer_name)
            
            if algo_mask is None:
                print(f"  {layer_name}: 算法输出中未找到")
                results.append({
                    'layer': layer_name,
                    'iou': None,
                    'dice': None,
                    'precision': None,
                    'recall': None,
                    'status': 'algo_missing'
                })
                continue
            
            if gt_mask is None:
                print(f"  {layer_name}: Ground Truth中未找到")
                results.append({
                    'layer': layer_name,
                    'iou': None,
                    'dice': None,
                    'precision': None,
                    'recall': None,
                    'status': 'gt_missing'
                })
                continue
            
            # 计算各指标
            iou = self.calculate_iou(algo_mask, gt_mask)
            dice = self.calculate_dice(algo_mask, gt_mask)
            
            # Precision和Recall
            tp = np.logical_and(algo_mask > 0, gt_mask > 0).sum()
            fp = np.logical_and(algo_mask > 0, gt_mask == 0).sum()
            fn = np.logical_and(algo_mask == 0, gt_mask > 0).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            print(f"  {layer_name}: IoU={iou:.4f}, Dice={dice:.4f}, "
                  f"Precision={precision:.4f}, Recall={recall:.4f}")
            
            results.append({
                'layer': layer_name,
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall,
                'status': 'ok'
            })
        
        # 计算平均值
        valid_results = [r for r in results if r['status'] == 'ok']
        if valid_results:
            avg_iou = np.mean([r['iou'] for r in valid_results])
            avg_dice = np.mean([r['dice'] for r in valid_results])
            print(f"\n  平均 IoU: {avg_iou:.4f}")
            print(f"  平均 Dice: {avg_dice:.4f}")
        
        return pd.DataFrame(results)
    
    def extract_layer_boundaries(self, layers_dict):
        """从分层mask中提取层边界线"""
        boundaries = {}
        h, w = list(layers_dict.values())[0].shape[:2]
        
        # 合并所有层的mask
        all_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in layers_dict.values():
            all_mask = cv2.bitwise_or(all_mask, mask)
        
        # 对每一层，找到其上边界和下边界
        for layer_name, mask in layers_dict.items():
            if mask is None:
                continue
            
            # 找边界：对mask进行形态学梯度
            kernel = np.ones((3, 3), np.uint8)
            gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            
            # 找边界点的y坐标（每列的最小和最大y）
            top_boundary = []
            bottom_boundary = []
            
            for x in range(w):
                col = gradient[:, x]
                y_coords = np.where(col > 0)[0]
                if len(y_coords) > 0:
                    top_boundary.append((x, y_coords.min()))
                    bottom_boundary.append((x, y_coords.max()))
            
            boundaries[layer_name] = {
                'top': np.array(top_boundary) if top_boundary else None,
                'bottom': np.array(bottom_boundary) if bottom_boundary else None
            }
        
        return boundaries
    
    def calculate_boundary_distance(self, boundary1, boundary2):
        """计算两条边界线之间的平均距离"""
        if boundary1 is None or boundary2 is None:
            return None
        if len(boundary1) == 0 or len(boundary2) == 0:
            return None
        
        # 对每个x坐标，找到对应的y值并计算距离
        # 构建x -> y的映射
        b1_dict = {p[0]: p[1] for p in boundary1}
        b2_dict = {p[0]: p[1] for p in boundary2}
        
        common_x = set(b1_dict.keys()) & set(b2_dict.keys())
        
        if not common_x:
            return None
        
        distances = [abs(b1_dict[x] - b2_dict[x]) for x in common_x]
        return np.mean(distances)
    
    def calculate_boundary_metrics(self):
        """计算层边界的距离误差"""
        print("\n" + "="*50)
        print("计算层边界距离误差...")
        print("="*50)
        
        # 提取边界
        algo_boundaries = self.extract_layer_boundaries(self.algo_layers)
        gt_boundaries = self.extract_layer_boundaries(self.gt_layers)
        
        results = []
        
        for layer_name in self.standard_layers:
            algo_b = algo_boundaries.get(layer_name, {})
            gt_b = gt_boundaries.get(layer_name, {})
            
            # 上边界距离
            top_dist = self.calculate_boundary_distance(
                algo_b.get('top'), gt_b.get('top'))
            
            # 下边界距离
            bottom_dist = self.calculate_boundary_distance(
                algo_b.get('bottom'), gt_b.get('bottom'))
            
            avg_dist = None
            if top_dist is not None and bottom_dist is not None:
                avg_dist = (top_dist + bottom_dist) / 2
            elif top_dist is not None:
                avg_dist = top_dist
            elif bottom_dist is not None:
                avg_dist = bottom_dist
            
            print(f"  {layer_name}: 上边界误差={top_dist:.2f}px, "
                  f"下边界误差={bottom_dist:.2f}px, 平均={avg_dist:.2f}px" 
                  if avg_dist else f"  {layer_name}: 无法计算")
            
            results.append({
                'layer': layer_name,
                'top_boundary_error': top_dist,
                'bottom_boundary_error': bottom_dist,
                'avg_boundary_error': avg_dist
            })
        
        # 计算总体平均
        valid_errors = [r['avg_boundary_error'] for r in results 
                       if r['avg_boundary_error'] is not None]
        if valid_errors:
            print(f"\n  总体平均边界误差: {np.mean(valid_errors):.2f} pixels")
        
        return pd.DataFrame(results)
    
    def calculate_thickness(self, mask):
        """计算每列的层厚度"""
        h, w = mask.shape[:2]
        thicknesses = []
        
        for x in range(w):
            col = mask[:, x]
            y_coords = np.where(col > 0)[0]
            if len(y_coords) > 0:
                thickness = y_coords.max() - y_coords.min() + 1
                thicknesses.append(thickness)
        
        return np.array(thicknesses) if thicknesses else None
    
    def calculate_thickness_metrics(self):
        """计算各层厚度误差"""
        print("\n" + "="*50)
        print("计算各层厚度误差...")
        print("="*50)
        
        results = []
        
        for layer_name in self.standard_layers:
            algo_mask = self.algo_layers.get(layer_name)
            gt_mask = self.gt_layers.get(layer_name)
            
            if algo_mask is None or gt_mask is None:
                print(f"  {layer_name}: 数据缺失")
                results.append({
                    'layer': layer_name,
                    'algo_mean_thickness': None,
                    'gt_mean_thickness': None,
                    'thickness_error': None,
                    'thickness_error_percent': None
                })
                continue
            
            algo_thickness = self.calculate_thickness(algo_mask)
            gt_thickness = self.calculate_thickness(gt_mask)
            
            if algo_thickness is None or gt_thickness is None:
                print(f"  {layer_name}: 无法计算厚度")
                continue
            
            algo_mean = np.mean(algo_thickness)
            gt_mean = np.mean(gt_thickness)
            error = abs(algo_mean - gt_mean)
            error_percent = (error / gt_mean * 100) if gt_mean > 0 else 0
            
            print(f"  {layer_name}: 算法厚度={algo_mean:.1f}px, GT厚度={gt_mean:.1f}px, "
                  f"误差={error:.1f}px ({error_percent:.1f}%)")
            
            results.append({
                'layer': layer_name,
                'algo_mean_thickness': algo_mean,
                'gt_mean_thickness': gt_mean,
                'thickness_error': error,
                'thickness_error_percent': error_percent
            })
        
        # 计算平均误差
        valid_results = [r for r in results if r['thickness_error'] is not None]
        if valid_results:
            avg_error = np.mean([r['thickness_error'] for r in valid_results])
            avg_error_pct = np.mean([r['thickness_error_percent'] for r in valid_results])
            print(f"\n  平均厚度误差: {avg_error:.2f}px ({avg_error_pct:.2f}%)")
        
        return pd.DataFrame(results)
    
    def visualize_comparison(self, save_path=None):
        """可视化比较结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像对比
        axes[0, 0].imshow(cv2.cvtColor(self.algo_mask, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Algorithm Output')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(self.gt_mask, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # 差异图
        diff = cv2.absdiff(self.algo_mask, self.gt_mask)
        axes[0, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Difference')
        axes[0, 2].axis('off')
        
        # 各层对比
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]  # R, G, B, Y
        
        # 算法输出各层叠加
        h, w = self.algo_mask.shape[:2]
        algo_overlay = np.zeros((h, w, 3), dtype=np.float32)
        for i, layer_name in enumerate(self.standard_layers):
            if layer_name in self.algo_layers:
                mask = self.algo_layers[layer_name] > 0
                for c in range(3):
                    algo_overlay[:, :, c][mask] = colors[i % len(colors)][c]
        
        axes[1, 0].imshow(algo_overlay)
        axes[1, 0].set_title('Algorithm Layers (R=L1, G=L2/3, B=L4, Y=L5/6)')
        axes[1, 0].axis('off')
        
        # GT各层叠加
        gt_overlay = np.zeros((h, w, 3), dtype=np.float32)
        for i, layer_name in enumerate(self.standard_layers):
            if layer_name in self.gt_layers:
                mask = self.gt_layers[layer_name] > 0
                for c in range(3):
                    gt_overlay[:, :, c][mask] = colors[i % len(colors)][c]
        
        axes[1, 1].imshow(gt_overlay)
        axes[1, 1].set_title('GT Layers (R=L1, G=L2/3, B=L4, Y=L5/6)')
        axes[1, 1].axis('off')
        
        # 重叠区域
        overlap = np.zeros((h, w, 3), dtype=np.float32)
        for i, layer_name in enumerate(self.standard_layers):
            if layer_name in self.algo_layers and layer_name in self.gt_layers:
                algo_m = self.algo_layers[layer_name] > 0
                gt_m = self.gt_layers[layer_name] > 0
                
                # 重叠区域用绿色
                overlap_m = np.logical_and(algo_m, gt_m)
                # 仅算法有用红色
                algo_only = np.logical_and(algo_m, ~gt_m)
                # 仅GT有用蓝色
                gt_only = np.logical_and(~algo_m, gt_m)
                
                overlap[:, :, 1][overlap_m] = 0.5 + i * 0.1  # Green
                overlap[:, :, 0][algo_only] = 0.5 + i * 0.1   # Red
                overlap[:, :, 2][gt_only] = 0.5 + i * 0.1     # Blue
        
        axes[1, 2].imshow(overlap)
        axes[1, 2].set_title('Overlap (G=match, R=algo only, B=GT only)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"\n可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "#"*60)
        print("# 分层结果验证分析")
        print("#"*60)
        
        # 加载和解析图像
        self.load_images()
        self.parse_masks_by_color()
        
        # 计算各项指标
        overlap_df = self.calculate_overlap_metrics()
        boundary_df = self.calculate_boundary_metrics()
        thickness_df = self.calculate_thickness_metrics()
        
        # 合并结果
        results_df = overlap_df.merge(boundary_df, on='layer', how='outer')
        results_df = results_df.merge(thickness_df, on='layer', how='outer')
        
        # 保存结果
        results_path = self.output_dir / "validation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n验证结果已保存到: {results_path}")
        
        # 可视化
        vis_path = self.output_dir / "validation_comparison.png"
        self.visualize_comparison(save_path=str(vis_path))
        
        # 打印总结
        print("\n" + "="*60)
        print("验证结果总结")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df


def main():
    """主函数"""
    print("="*60)
    print("分层结果验证工具")
    print("="*60)
    print("\n使用固定颜色映射:")
    print("Ground Truth颜色 → 层级")
    print("  BGR(25,28,252) → L1")
    print("  BGR(18,126,126) → L2/3")
    print("  BGR(255,255,76) → L4")
    print("  BGR(127,127,248) → L5/6")
    print("\n算法输出颜色 → 层级")
    print("  BGR(100,100,248) → L1")
    print("  BGR(109,254,118) → L2/3")
    print("  BGR(252,104,103) → L4")
    print("  BGR(109,254,254) → L5/6")
    print("\n计算指标:")
    print("  1. IoU (交并比)")
    print("  2. Dice系数")
    print("  3. Precision & Recall")
    print("  4. 层边界距离误差")
    print("  5. 层厚度误差")
    print("="*60)
    
    analyzer = LayerResultAnalyzer(
        output_dir="output",
        input_dir="input"
    )
    
    results = analyzer.run_full_analysis()
    
    print("\n✓ 分析完成!")
    print(f"结果已保存到: output/validation_results.csv")
    print(f"可视化已保存到: output/validation_comparison.png")


if __name__ == "__main__":
    main()
