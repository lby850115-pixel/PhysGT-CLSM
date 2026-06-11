// load_pred_rois.ijm
// ==================
// 将 MitoSegNet 预测标签（uint16 实例标签）批量转换为 ROI Manager 条目，
// 同时打开对应的 8-bit 原始图作为参考底图，供用户人工校正。
//
// 使用方法：
//   Fiji: Plugins > Macros > Run > load_pred_rois.ijm
//   修改下方 BASE_DIR 和 STEM 两个变量后运行。
//
// 操作完成后运行 fill_rois_as_labels.ijm 生成 GT mask。

// ── 用户配置区（每次标注一张图前修改这里）────────────────────────────────
BASE_DIR = "D:/My research data/mito_validation/data/semi_auto/";
STEM     = "Series086-HELA-2uM-6h-mito-lyso-ok_z0_ch02";
// ─────────────────────────────────────────────────────────────────────────

pred_path = BASE_DIR + STEM + "_pred_labels.tif";
raw_path  = BASE_DIR + STEM + "_raw8.tif";

// 1. 清空 ROI Manager
roiManager("reset");

// 2. 打开预测标签图，Analyze Particles 将每个实例转换为 ROI
open(pred_path);
rename("pred_labels");
setThreshold(1, 65535);
run("Analyze Particles...", "size=0-Infinity show=Nothing add in_situ");
selectWindow("pred_labels");
close();

// 3. 打开 8-bit 原始图作为底层参考
open(raw_path);
rename("raw8_" + STEM);
run("Enhance Contrast", "saturated=0.35");  // 自动调整亮度便于观察

// 4. 操作说明弹窗
n_roi = roiManager("count");
msg = "已加载 " + n_roi + " 个预测 ROI 到 ROI Manager。\n\n";
msg += "请进行以下操作：\n";
msg += "  删除假阳性  ：ROI Manager 中选中 → Delete\n";
msg += "  补充漏检    ：Polygon/Freehand 圈选 → 按 T\n";
msg += "  调整边界    ：双击 ROI → 拖拽控制点\n\n";
msg += "【衍射极限原则】\n";
msg += "  ✓ 只标注边界清晰可分辨的线粒体\n";
msg += "  ✗ 直径 < 2px 的模糊斑点不标注\n";
msg += "  ~ 接触线粒体：能分辨间隙则分开，否则合并为一个 ROI\n\n";
msg += "完成校正后，运行 fill_rois_as_labels.ijm 生成 GT mask。";
showMessage("半自动标注 — " + STEM, msg);
