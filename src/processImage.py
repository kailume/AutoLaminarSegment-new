import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os

# 默认输入输出路径
input_path = "input"
output_path = "output"
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)


def resolve_image_path(directory, filename):
    base = os.path.join(directory, filename)
    if os.path.exists(base):
        return base

    stem, suffix = os.path.splitext(filename)
    if not suffix:
        stem = filename
    for ext in IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = os.path.join(directory, stem + candidate_ext)
            if os.path.exists(candidate):
                return candidate
    return base

def croptissue(image):

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # CLAHE 自适应直方图均衡化
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # enhanced_image = clahe.apply(blurred_image)

    enhanced_image = blurred_image.copy()

    # cv2.imwrite("debug/enhanced_image.png", enhanced_image)

    # Otsu算法得到阈值
    otsu_threshold, _ = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold *= 1.5
    _, thresholded_image = cv2.threshold(enhanced_image, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

    # # 自适应阈值分割
    # # 计算图像大小
    # height, width = enhanced_image.shape
    # # 根据图像尺寸调整块大小
    # block_size = max(15, (min(height, width) // 20) | 1)
    # thresholded_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                           cv2.THRESH_BINARY, block_size, 2)

    # # 使用层级分水岭算法分割
    # distance_transform = cv2.distanceTransform(enhanced_image, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(enhanced_image, sure_fg)
    # _, markers = cv2.connectedComponents(sure_fg)
    # markers = markers + 1
    # markers[unknown == 255] = 0
    # markers = cv2.watershed(cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR), markers)
    # thresholded_image = np.zeros_like(enhanced_image, dtype=np.uint8)
    # thresholded_image[markers > 1] = 255
    # thresholded_image = cv2.bitwise_not(thresholded_image)

    # # 区域生长法分割
    # height, width = enhanced_image.shape
    # seed_point = (width // 2, height // 2)  # 图像中心点作为种子点
    # mask = np.zeros((height + 2, width + 2), np.uint8)
    # flood_fill_flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    # cv2.floodFill(enhanced_image, mask, seed_point, 255, loDiff=10, upDiff=10, flags=flood_fill_flags)
    # thresholded_image = mask[1:-1, 1:-1]  # 去掉多余的边框

    # thresholded_image = cv2.bitwise_not(thresholded_image)

    # cv2.imwrite("temp/thresholded_image.png", thresholded_image)

    # 形态学变换
    kernel_size =99
    # kernel_size = max(51, min(image.shape[0], image.shape[1]) // 20 | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # cv2.imwrite("debug/closed_image.png", closed_image)

    # 寻找最大联通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)
    if num_labels > 1:  
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = np.zeros_like(labels, dtype=np.uint8)
        largest_component[labels == largest_label] = 255
    tissue_mask = largest_component.copy()

    # # 对tissue_mask进行膨胀，确保覆盖整个组织区域
    # kernel_size = 15
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # tissue_mask = cv2.dilate(tissue_mask, dilate_kernel, iterations=1)
    # # cv2.imwrite("debug/tissue_mask_dilated.png", tissue_mask)

    # 提取组织外部轮廓
    # tissue_contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue_contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # tissue_contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(tissue_contour_image, tissue_contours, -1, (0, 255, 0), 5)
    # cv2.imwrite("debug/tissue_contours.png", tissue_contour_image)

    # 填充组织内部区域
    cv2.drawContours(tissue_mask, tissue_contours, -1, 255, thickness=cv2.FILLED)
    # cv2.imwrite("debug/tissue_mask_filled.png", tissue_mask)

    return tissue_mask, tissue_contours

def cropwhite(image, tissue_mask, tissue_image):

    # 统计tissue_mask对应区域的灰度直方图
    hist = cv2.calcHist([image], [0], tissue_mask, [256], [0, 256])
    # 根据灰度直方图计算otsu阈值
    total_pixels = np.sum(hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.dot(np.arange(256), hist.flatten())
    otsu_threshold_custom = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            otsu_threshold_custom = i

    otsu_threshold_custom *= 0.25

    # 根据otsu_threshold_custom分割tissue_mask
    _, result_mask = cv2.threshold(tissue_image, otsu_threshold_custom, 255, cv2.THRESH_BINARY_INV)
    white_mask = result_mask.copy()
    white_mask = cv2.bitwise_and(white_mask, tissue_mask)

    # 形态学变换
    kernel_size = 51
    # kernel_size = max(51, min(image.shape[0], image.shape[1]) // 20 | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # cv2.imwrite("debug/white_mask.png", white_mask)

    # 计算white_mask的最大联通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    if num_labels > 1:  
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = np.zeros_like(labels, dtype=np.uint8)
        largest_component[labels == largest_label] = 255
    white_mask = largest_component.copy()
    # 膨胀white_mask以覆盖更多区域
    kernel_size = 25
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    white_mask = cv2.dilate(white_mask, dilate_kernel, iterations=1)

    # cv2.imwrite("debug/white_mask_largest.png", white_mask)

    # 提取白质轮廓
    # white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    white_contour_image = np.ones_like(white_mask, dtype=np.uint8) * 255
    white_contour_image = cv2.cvtColor(white_contour_image, cv2.COLOR_GRAY2BGR)
    # white_contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(white_contour_image, white_contours, -1, (0, 0, 255), 5)
    cv2.imwrite("debug/white_contours.png", white_contour_image)

    # 填充白质内部区域
    cv2.drawContours(white_mask, white_contours, -1, 255, thickness=cv2.FILLED)

    return white_mask, white_contours

def extractgray(tissue_mask, white_mask):
    gray_mask = cv2.bitwise_and(tissue_mask, cv2.bitwise_not(white_mask))

    # 形态学变换去除边缘噪声
    kernel_size = 79
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 保留最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)
    if num_labels > 1:  
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = np.zeros_like(labels, dtype=np.uint8)
        largest_component[labels == largest_label] = 255
    gray_mask = largest_component.copy()

    # 提取灰质轮廓
    # gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 在空白上绘制灰质轮廓并保存
    gray_contour_image = np.ones_like(gray_mask, dtype=np.uint8) * 255
    gray_contour_image = cv2.cvtColor(gray_contour_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(gray_contour_image, gray_contours, -1, (255, 0, 0), 5)
    cv2.imwrite("debug/gray_contours.png", gray_contour_image)


    return gray_mask, gray_contours

def computeContours(white_contours, gray_contours, issave=True):

    # 从 gray_contours 中移除靠近 white_contours 的点
    # 距离阈值（像素），小于该值的灰质点会被删除
    distance_threshold = 225
    outer_contours = []
    inner_contours = []
    if len(white_contours) == 0:
        outer_contours = gray_contours
    else:
        for cnt in gray_contours:
            kept_pts = []
            removed_pts = []
            for pt in cnt:
                x, y = int(pt[0][0]), int(pt[0][1])
                min_dist = float('inf')
                for wcnt in white_contours:
                    # pointPolygonTest 返回带符号距离（内部为正），取绝对值表示到轮廓的距离
                    dist = abs(cv2.pointPolygonTest(wcnt, (x, y), True))
                    if dist < min_dist:
                        min_dist = dist
                    # 若已经在白质轮廓内部或非常近，则可以提前判定删除
                    if min_dist <= 0:
                        removed_pts.append([[x, y]])
                        break
                # 仅保留距离所有白质轮廓都超过阈值的点
                if min_dist > distance_threshold:
                    kept_pts.append([[x, y]])
                else:
                    removed_pts.append([[x, y]])
            # 保证多边形至少有 3 个点
            if len(kept_pts) >= 3:
                outer_contours.append(np.array(kept_pts, dtype=np.int32))
            if len(removed_pts) > 0:
                inner_contours.append(np.array(removed_pts, dtype=np.int32))

    if issave:
        with open(f"{output_path}\\GM.csv", "w") as f:
            f.write("x,y\n")
            for contour in outer_contours:
                for point in contour:
                    x, y = point[0]
                    f.write(f"{x},{y}\n")
        with open(f"{output_path}\\WM.csv", "w") as f:
            f.write("x,y\n")
            for contour in inner_contours:
                for point in contour:
                    x, y = point[0]
                    f.write(f"{x},{y}\n")
    
    return outer_contours, inner_contours

def subtractBackground(image, src_size=51):
    """
    使用 Sliding Paraboloid 算法提取背景并从图像中减去。
    使用分块并行处理加速计算。
    
    参数:
        image: 输入的灰度图像
        src_size: 滚动球/抛物面的半径（像素）
    
    返回:
        背景减除后的图像
    """
    height, width = image.shape
    img_float = image.astype(np.float64)
    radius = src_size
    
    # 计算最大偏移量
    max_offset = int(np.ceil(np.sqrt(2 * radius * 255)))
    
    def rolling_paraboloid_1d(line, radius, max_offset):
        """对一维数组应用滚动抛物面算法（向量化优化版本）"""
        n = len(line)
        max_off = min(max_offset, n)
        result = np.full(n, -np.inf)
        
        # 预计算偏移量对应的抛物面衰减
        offsets = np.arange(-max_off, max_off + 1)
        paraboloid_decay = (offsets ** 2) / (2.0 * radius)
        
        for i in range(n):
            start = max(0, i - max_off)
            end = min(n, i + max_off + 1)
            
            # 计算当前位置对应的偏移范围
            local_start = start - i + max_off
            local_end = end - i + max_off
            
            # 向量化计算
            heights = line[start:end] - paraboloid_decay[local_start:local_end]
            result[i] = np.max(heights)
        
        return result
    
    def process_rows_chunk(args):
        """处理一块行"""
        start_row, end_row, img_chunk, radius, max_offset = args
        chunk_height = end_row - start_row
        result_chunk = np.zeros((chunk_height, img_chunk.shape[1]), dtype=np.float64)
        for i in range(chunk_height):
            result_chunk[i, :] = rolling_paraboloid_1d(img_chunk[i, :], radius, max_offset)
        return start_row, result_chunk
    
    def process_cols_chunk(args):
        """处理一块列"""
        start_col, end_col, img_chunk, radius, max_offset = args
        chunk_width = end_col - start_col
        result_chunk = np.zeros((img_chunk.shape[0], chunk_width), dtype=np.float64)
        for i in range(chunk_width):
            result_chunk[:, i] = rolling_paraboloid_1d(img_chunk[:, i], radius, max_offset)
        return start_col, result_chunk
    
    # 确定并行线程数
    num_workers = min(multiprocessing.cpu_count(), 8)
    
    # ===== 水平方向滚动（按行分块） =====
    background = np.zeros_like(img_float, dtype=np.float64)
    chunk_size = max(1, height // num_workers)
    
    row_tasks = []
    for i in range(0, height, chunk_size):
        end_row = min(i + chunk_size, height)
        row_tasks.append((i, end_row, img_float[i:end_row, :], radius, max_offset))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_rows_chunk, row_tasks))
    
    for start_row, result_chunk in results:
        end_row = start_row + result_chunk.shape[0]
        background[start_row:end_row, :] = result_chunk
    
    # ===== 垂直方向滚动（按列分块） =====
    temp = np.zeros_like(background, dtype=np.float64)
    chunk_size = max(1, width // num_workers)
    
    col_tasks = []
    for i in range(0, width, chunk_size):
        end_col = min(i + chunk_size, width)
        col_tasks.append((i, end_col, background[:, i:end_col], radius, max_offset))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_cols_chunk, col_tasks))
    
    for start_col, result_chunk in results:
        end_col = start_col + result_chunk.shape[1]
        temp[:, start_col:end_col] = result_chunk
    
    background = temp
    
    # 确保背景不超过原图
    background = np.minimum(background, img_float)
    
    # 从原图减去背景
    subtracted = img_float - background
    
    # 归一化到 0-255
    subtracted = np.clip(subtracted, 0, 255)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    subtracted = subtracted.astype(np.uint8)
    
    subtracted = cv2.convertScaleAbs(subtracted, alpha=1.5, beta=0)

    return subtracted

def splitContour(cnt, max_gap):
    """
    将单个轮廓按相邻点间距拆分为若干子段，若相邻点距离大于 max_gap 则断开。
    返回值为子轮廓列表，每个子轮廓为 shape=(N,1,2) 的 int32 numpy 数组。
    """
    segments = []
    if cnt is None or len(cnt) == 0:
        return segments
    current = [cnt[0][0].tolist()]  # 以 [x,y] 形式存储点
    for i in range(1, len(cnt)):
        p_prev = cnt[i-1][0]
        p = cnt[i][0]
        dist = np.hypot(float(p[0] - p_prev[0]), float(p[1] - p_prev[1]))
        if dist <= max_gap:
            current.append(p.tolist())
        else:
            if len(current) >= 2:
                seg = np.array(current, dtype=np.int32).reshape(-1, 1, 2)
                segments.append(seg)
            current = [p.tolist()]
    if len(current) >= 2:
        seg = np.array(current, dtype=np.int32).reshape(-1, 1, 2)
        segments.append(seg)
    return segments


if __name__ == "__main__":
    image = cv2.imread(resolve_image_path(input_path, "4x.png"), cv2.IMREAD_GRAYSCALE)
    # fluorescence_image = cv2.imread(f"{input_path}\\40x.png", cv2.IMREAD_GRAYSCALE)

    # fluorescence_image = subtractBackground(fluorescence_image, src_size=51)
    # cv2.imwrite("fluorescence_subtracted.png", fluorescence_image)


    # 裁剪组织区域
    tissue_mask, tissue_contours = croptissue(image)
    tissue_image = cv2.bitwise_and(image, image, mask=tissue_mask)
    cv2.imwrite(f"{output_path}\\tissueImage.png", tissue_image)

    # exit()

    # 裁剪白质区域
    white_mask, white_contours = cropwhite(image, tissue_mask, tissue_image)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    cv2.imwrite(f"{output_path}\\whiteMask.png", white_mask)
    cv2.imwrite(f"{output_path}\\whiteImage.png", white_image)

    # exit()


    # 计算灰质区域
    gray_mask, gray_contours = extractgray(tissue_mask, white_mask)
    gray_image = cv2.bitwise_and(image, image, mask=gray_mask)
    cv2.imwrite(f"{output_path}\\grayImage.png", gray_image)
    cv2.imwrite(f"{output_path}\\grayMask.png", gray_mask)

    # exit()

    # 计算荧光场灰质区域
    # gray_fluo_image = cv2.bitwise_and(fluorescence_image, fluorescence_image, mask=gray_mask)
    # cv2.imwrite(f"{output_path}\\grayFluoImage.png", gray_fluo_image)
    # 计算边界线
    outer_contours, inner_contours = computeContours(white_contours, gray_contours, issave=True)

    # 在空白图像上绘制过滤后的灰质轮廓为绿色
    # empty_image = np.ones_like(image, dtype=np.uint8) * 255
    empty_image = image.copy()
    empty_image = cv2.cvtColor(empty_image, cv2.COLOR_GRAY2BGR)

    # 在空白图像上绘制outer_contours和 inner_contours为绿色和红色散点
    for cnt in outer_contours:
        for point in cnt:
            x, y = point[0]
            # 如果在边缘上则跳过
            if x <=0 or y <=0 or x >= image.shape[1]-1 or y >= image.shape[0]-1:
                continue
            cv2.circle(empty_image, (x, y), radius=3, color=(255, 0, 255), thickness=-1)
    for cnt in inner_contours:
        for point in cnt:
            x, y = point[0]
            # 如果在边缘上则跳过
            if x <=0 or y <=0 or x >= image.shape[1]-1 or y >= image.shape[0]-1:
                continue
            cv2.circle(empty_image, (x, y), radius=3, color=(255, 255, 0), thickness=-1)
    cv2.imwrite(f"{output_path}\\OuterInnerPoints.png", empty_image)
