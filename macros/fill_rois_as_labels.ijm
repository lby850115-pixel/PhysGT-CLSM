// fill_rois_as_labels.ijm
// =======================
// 将 ROI Manager 中所有 ROI 填充为唯一整数标签（1..N），
// 生成 uint16 实例标签 TIF 并保存到 data/gt_masks/。
//
// 前置条件：
//   - ROI Manager 中已有校正后的 ROI（由 load_pred_rois.ijm 加载并人工校正）
//   - 当前活动图像应为 raw8 底图（尺寸 1024×1024）
//
// 使用方法：
//   在 load_pred_rois.ijm 校正完成后运行此宏。
//   修改下方 GT_DIR 和 STEM 两个变量后运行。
//
// 使用方法：
//   Fiji: Plugins > Macros > Run > fill_rois_as_labels.ijm

// ── 用户配置区（与 load_pred_rois.ijm 保持一致）──────────────────────────
GT_DIR = "D:/My research data/mito_validation/data/gt_masks/";
STEM   = "Series086-HELA-2uM-6h-mito-lyso-ok_z0_ch02";
// ─────────────────────────────────────────────────────────────────────────

n_roi = roiManager("count");
if (n_roi == 0) {
    showMessage("错误", "ROI Manager 为空！\n请先运行 load_pred_rois.ijm 并完成校正。");
    exit();
}

// 获取当前图像尺寸（从 raw8 底图读取）
getDimensions(W, H, ch, sl, fr);

// 创建新的 16-bit 黑色图像作为标签画布
newImage("GT_mask", "16-bit black", W, H, 1);
gt_id = getImageID();

// 逐 ROI 填充唯一整数标签
for (i = 0; i < n_roi; i++) {
    selectImage(gt_id);
    roiManager("Select", i);
    setColor(i + 1);    // 标签从 1 开始，0 为背景
    fill();
}

// 去除选择框
run("Select None");

// 保存 GT mask
out_path = GT_DIR + STEM + "_mask_GT.tif";
File.makeDirectory(GT_DIR);
saveAs("Tiff", out_path);

// 完成提示
showMessage("GT 保存完成",
    "已保存 " + n_roi + " 个实例标签。\n\n" +
    "输出文件：\n" + out_path + "\n\n" +
    "下一步：\n" +
    "  python check_mito_gt.py\n" +
    "  python 02_evaluate_all.py");
