1. cellpose 分割得到 cell_cendroids.csv
2. draw_boundaries 得到 GM.csv、WM.csv、mask.png
3. dapi.png原图、cell_centroids.csv、GM.csv、WM.csv、mask.png移动到input文件夹
4. 运行run_pipeline.py：python .\run_pipeline.py input\P00185-T001-R001-S014-B1-DAPI-MaxIP_RGB --compact # --compact 对大图进行压缩（10：1）

fast推理：python segment_large_mage_fast.py <image.tif> --tile-size 4096 --no-tta
# 更大的tilesize + 取消多尺度重采样（边界精度下降）

完整pipeline（4xbright图像利用）：run_auto_pipeline.py