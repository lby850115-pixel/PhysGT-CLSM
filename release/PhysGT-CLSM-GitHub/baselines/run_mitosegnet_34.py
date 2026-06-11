"""run_mitosegnet_34.py — MitoSegNet inference on all 34 CLSM images."""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.ndimage import label as nd_label

ROOT      = Path('/home/zhishi/LBY_CELL')
DATA_DIRS = {
    'HELA':  ROOT / 'data' / 'mito_34' / 'HELA',
    'BXPC3': ROOT / 'data' / 'mito_34' / 'BXPC3',
    'MCF7':  ROOT / 'data' / 'mito_34' / 'MCF7',
}
CKPT      = ROOT / 'checkpoints' / 'mitosegnet' / 'mitosegnet_best.pt'
OUT_PRED  = ROOT / 'results' / 'predictions_34' / 'mitosegnet'
OUT_FIG   = ROOT / 'figures'  / 'predictions_34' / 'mitosegnet'
OUT_PRED.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
MIN_AREA  = 10
TILE      = 256
OVERLAP   = 32
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.pool  = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        return self.pool(x), x


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = ConvBnRelu(out_ch * 2, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv2(self.conv1(x))


class MitoSegNet(nn.Module):
    def __init__(self, f=32):
        super().__init__()
        self.enc1 = EncBlock(1,   f)
        self.enc2 = EncBlock(f,   f*2)
        self.enc3 = EncBlock(f*2, f*4)
        self.enc4 = EncBlock(f*4, f*8)
        self.bottleneck = nn.Sequential(ConvBnRelu(f*8, f*16), ConvBnRelu(f*16, f*16))
        self.dec4 = DecBlock(f*16, f*8)
        self.dec3 = DecBlock(f*8,  f*4)
        self.dec2 = DecBlock(f*4,  f*2)
        self.dec1 = DecBlock(f*2,  f)
        self.out  = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        x, s1 = self.enc1(x); x, s2 = self.enc2(x)
        x, s3 = self.enc3(x); x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4); x = self.dec3(x, s3)
        x = self.dec2(x, s2); x = self.dec1(x, s1)
        return torch.sigmoid(self.out(x))


def load_mito(fp):
    img = tifffile.imread(fp)
    if img.ndim == 3:
        ch = int(np.argmax([img[:, :, c].mean() for c in range(img.shape[2])]))
        return img[:, :, ch].astype(np.float32)
    return img.astype(np.float32)


def predict_tiled(model, img):
    stride  = TILE - OVERLAP * 2
    img_pad = np.pad(img, OVERLAP, mode='reflect')
    Hp, Wp  = img_pad.shape
    psum = np.zeros((Hp, Wp), np.float32)
    pcnt = np.zeros((Hp, Wp), np.float32)
    ys = list(range(0, Hp - TILE + 1, stride))
    xs = list(range(0, Wp - TILE + 1, stride))
    if ys and ys[-1] + TILE < Hp: ys.append(Hp - TILE)
    if xs and xs[-1] + TILE < Wp: xs.append(Wp - TILE)
    tiles = [(y, x) for y in ys for x in xs]
    with torch.no_grad():
        for ti, (y, x) in enumerate(tiles):
            t = img_pad[y:y+TILE, x:x+TILE]
            lo, hi = np.percentile(t, [1, 99.8])
            tn = np.clip((t - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
            prob = model(torch.from_numpy(tn[None, None]).to(DEVICE))[0, 0].cpu().numpy()
            psum[y:y+TILE, x:x+TILE] += prob
            pcnt[y:y+TILE, x:x+TILE] += 1
            print(f"    tile {ti+1}/{len(tiles)}", flush=True)
    H, W = img.shape
    return psum[OVERLAP:OVERLAP+H, OVERLAP:OVERLAP+W] / np.maximum(
        pcnt[OVERLAP:OVERLAP+H, OVERLAP:OVERLAP+W], 1)


def to_instances(prob):
    binary = (prob >= THRESHOLD).astype(np.uint8)
    lab, _ = nd_label(binary)
    res = np.zeros_like(lab, dtype=np.uint16)
    nid = 1
    for iid in range(1, lab.max() + 1):
        if (lab == iid).sum() >= MIN_AREA:
            res[lab == iid] = nid; nid += 1
    return res


def main():
    print(f"Device: {DEVICE}")
    model = MitoSegNet(f=32).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f"  Loaded {CKPT}")

    all_images = [(fp, ct) for ct, d in DATA_DIRS.items() if d.exists()
                  for fp in sorted(d.glob('*.tif'))]
    print(f"Total images: {len(all_images)}")

    for fp, ct in all_images:
        out_tif = OUT_PRED / fp.name
        if out_tif.exists() and tifffile.imread(out_tif).max() > 0:
            print(f"  [skip] {fp.name}"); continue

        print(f"\n[{fp.name}] ({ct})", flush=True)
        mito    = load_mito(fp)
        prob    = predict_tiled(model, mito)
        labeled = to_instances(prob)
        n       = int(labeled.max())
        tifffile.imwrite(out_tif, labeled)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        lo, hi = np.percentile(mito, [1, 99.5])
        mn = np.clip((mito - lo) / (hi - lo + 1e-9), 0, 1)
        axes[0].imshow(mn, cmap='gray'); axes[0].set_title('Mito'); axes[0].axis('off')
        cmap = plt.cm.get_cmap('tab20', max(n, 1))
        disp = np.zeros((*labeled.shape, 4))
        for iid in range(1, n + 1): disp[labeled == iid] = cmap(iid % 20)
        axes[1].imshow(mn, cmap='gray'); axes[1].imshow(disp, alpha=0.6)
        axes[1].set_title(f'n={n}'); axes[1].axis('off')
        plt.suptitle(f'{fp.stem} | MitoSegNet | n={n}', fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_FIG / f'{fp.stem}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  n={n}", flush=True)

    print(f"\nDone. Predictions -> {OUT_PRED}")


if __name__ == '__main__':
    main()
