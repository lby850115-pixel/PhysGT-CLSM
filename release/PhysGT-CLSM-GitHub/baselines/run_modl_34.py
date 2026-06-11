"""run_modl_34.py — MoDL inference on all 34 CLSM images (server)."""
from __future__ import annotations
import csv, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import h5py
import tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label, distance_transform_edt, maximum_filter
from skimage.measure import regionprops

ROOT      = Path('/home/zhishi/LBY_CELL')
DATA_DIRS = {
    'HELA':  ROOT / 'data' / 'mito_34' / 'HELA',
    'BXPC3': ROOT / 'data' / 'mito_34' / 'BXPC3',
    'MCF7':  ROOT / 'data' / 'mito_34' / 'MCF7',
}
MODEL_PATH = ROOT / 'MoDL' / 'model' / 'U-RNet+.hdf5'
OUT_PRED   = ROOT / 'results' / 'predictions_34' / 'modl'
OUT_FIG    = ROOT / 'figures'  / 'predictions_34' / 'modl'
OUT_PRED.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

THRESHOLD    = 0.9
MIN_DISTANCE = 8
MIN_AREA     = 10
TILE_SIZE    = 512
OVERLAP      = 64


class NumpyModel:
    def __init__(self, path):
        self.w = {}
        with h5py.File(path, 'r') as f:
            f['model_weights'].visititems(
                lambda n, o: self.w.update({n: np.array(o, np.float32)})
                if isinstance(o, h5py.Dataset) else None)
        print(f"  Loaded {len(self.w)} weight arrays")

    def _k(self, n): return self.w[f"{n}/{n}/kernel:0"]
    def _b(self, n): return self.w[f"{n}/{n}/bias:0"]
    def _kt(self, n):
        k = self.w[f"{n}/{n}/kernel:0"]; return k.transpose(0,1,3,2)
    def _dk(self, n): return self.w[f"{n}/{n}/kernel:0"]
    def _db(self, n): return self.w[f"{n}/{n}/bias:0"]

    @staticmethod
    def _conv(x, k, b, pad='same'):
        kH,kW,ic,oc = k.shape; H,W = x.shape[1],x.shape[2]
        ph,pw = (kH//2,kW//2) if pad=='same' else (0,0)
        xp = np.pad(x[0],((ph,ph),(pw,pw),(0,0)))
        Hp,Wp = xp.shape[0]-kH+1, xp.shape[1]-kW+1
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(Hp,Wp,kH,kW,ic),
            strides=(xp.strides[0],xp.strides[1],xp.strides[0],xp.strides[1],xp.strides[2])
        ).reshape(Hp*Wp, kH*kW*ic)
        return (cols @ k.reshape(kH*kW*ic,oc) + b).reshape(Hp,Wp,oc)[np.newaxis]

    @staticmethod
    def _relu(x): return np.maximum(x,0)
    @staticmethod
    def _sig(x): return 1/(1+np.exp(-np.clip(x,-88,88)))
    @staticmethod
    def _pool(x):
        x2=x[:,:x.shape[1]//2*2,:x.shape[2]//2*2,:]
        return np.maximum(np.maximum(x2[:,0::2,0::2,:],x2[:,0::2,1::2,:]),
                          np.maximum(x2[:,1::2,0::2,:],x2[:,1::2,1::2,:]))
    @staticmethod
    def _up(x): return np.repeat(np.repeat(x,2,axis=1),2,axis=2)

    def C(self,n): return lambda x: self._relu(self._conv(x,self._k(n),self._b(n)))
    def Cv(self,n): return lambda x: self._relu(self._conv(x,self._k(n),self._b(n),'valid'))
    def Cl(self,n): return lambda x: self._conv(x,self._k(n),self._b(n),'valid')

    def _cb(self,x,c1,c2,c3,cs):
        m=self._relu(self._conv(x,self._k(c1),self._b(c1),'valid'))
        m=self._relu(self._pool(self._conv(m,self._k(c2),self._b(c2),'same')))
        m=self._relu(self._conv(m,self._k(c3),self._b(c3),'valid'))
        s=self._relu(self._pool(self._conv(x,self._k(cs),self._b(cs),'valid')))
        return self._relu(m+s)

    def _ib(self,x,c1,c2,c3):
        m=self._relu(self._conv(x,self._k(c1),self._b(c1),'valid'))
        m=self._relu(self._conv(m,self._k(c2),self._b(c2),'same'))
        m=self._relu(self._conv(m,self._k(c3),self._b(c3),'valid'))
        return self._relu(m+x)

    def _cbu(self,x,c1,c2,c3,cs):
        m=self._relu(self._conv(x,self._k(c1),self._b(c1),'valid'))
        m=self._relu(self._conv(m,self._k(c2),self._b(c2),'same'))
        m=self._relu(self._conv(m,self._k(c3),self._b(c3),'valid'))
        s=self._relu(self._conv(x,self._k(cs),self._b(cs),'valid'))
        return self._relu(m+s)

    def _dec_up(self,x,skip,ct1,ct2):
        u=self._relu(self._conv(self._up(x),self._kt(ct1),self._b(ct1),'same'))
        u=self._relu(self._conv(self._up(u),self._kt(ct2),self._b(ct2),'same'))
        h,w=min(u.shape[1],skip.shape[1]),min(u.shape[2],skip.shape[2])
        return np.concatenate([skip[:,:h,:w,:],u[:,:h,:w,:]],axis=-1)

    def _cbam(self,e1):
        mp=e1.max(axis=(1,2)); ap=e1.mean(axis=(1,2))
        mp=self._relu(mp@self._dk('dense')+self._db('dense'))
        mp=mp@self._dk('dense_2')+self._db('dense_2')
        ap=self._relu(ap@self._dk('dense_1')+self._db('dense_1'))
        ap=ap@self._dk('dense_3')+self._db('dense_3')
        ca=self._sig(mp+ap)[:,np.newaxis,np.newaxis,:]
        xca=e1*ca
        sp_in=np.concatenate([xca.max(-1,keepdims=True),xca.mean(-1,keepdims=True)],axis=-1)
        sp=self._sig(self._conv(sp_in,self._k('conv2d_86'),self._b('conv2d_86'),'same'))
        return xca*sp

    def predict(self,x):
        e1p=self._conv(x,self._k('conv2d'),self._b('conv2d'),'same')
        e1=self._relu(e1p)
        e1p2=self._conv(e1,self._k('conv2d_1'),self._b('conv2d_1'),'same')
        e1=self._relu(e1p2); p1=self._pool(e1)
        e2=self.C('conv2d_2')(p1); e2=self.C('conv2d_3')(e2)
        e2r=self._cb(e2,'conv2d_4','conv2d_5','conv2d_6','conv2d_7')
        e2r=self._ib(e2r,'conv2d_8','conv2d_9','conv2d_10')
        e2r=self._ib(e2r,'conv2d_11','conv2d_12','conv2d_13'); p2=self._pool(e2r)
        e3=self.C('conv2d_14')(p2); e3=self.C('conv2d_15')(e3)
        e3r=self._cb(e3,'conv2d_16','conv2d_17','conv2d_18','conv2d_19')
        e3r=self._ib(e3r,'conv2d_20','conv2d_21','conv2d_22')
        e3r=self._ib(e3r,'conv2d_23','conv2d_24','conv2d_25'); p3=self._pool(e3r)
        e4=self.C('conv2d_26')(p3); e4=self.C('conv2d_27')(e4)
        e4r=self._cb(e4,'conv2d_28','conv2d_29','conv2d_30','conv2d_31')
        e4r=self._ib(e4r,'conv2d_32','conv2d_33','conv2d_34')
        e4r=self._ib(e4r,'conv2d_35','conv2d_36','conv2d_37'); p4=self._pool(e4r)
        b=self.C('conv2d_38')(p4); b=self.C('conv2d_39')(b)
        br=self._cb(b,'conv2d_40','conv2d_41','conv2d_42','conv2d_43')
        br=self._ib(br,'conv2d_44','conv2d_45','conv2d_46')
        br=self._ib(br,'conv2d_47','conv2d_48','conv2d_49')
        cat4=self._dec_up(br,e4r,'conv2d_transpose','conv2d_transpose_1')
        d4=self.C('conv2d_50')(cat4); d4=self.C('conv2d_51')(d4)
        d4r=self._cbu(d4,'conv2d_52','conv2d_53','conv2d_54','conv2d_55')
        d4r=self._ib(d4r,'conv2d_56','conv2d_57','conv2d_58')
        d4r=self._ib(d4r,'conv2d_59','conv2d_60','conv2d_61')
        cat3=self._dec_up(d4r,e3r,'conv2d_transpose_2','conv2d_transpose_3')
        d3=self.C('conv2d_62')(cat3); d3=self.C('conv2d_63')(d3)
        d3r=self._cbu(d3,'conv2d_64','conv2d_65','conv2d_66','conv2d_67')
        d3r=self._ib(d3r,'conv2d_68','conv2d_69','conv2d_70')
        d3r=self._ib(d3r,'conv2d_71','conv2d_72','conv2d_73')
        cat2=self._dec_up(d3r,e2r,'conv2d_transpose_4','conv2d_transpose_5')
        d2=self.C('conv2d_74')(cat2); d2=self.C('conv2d_75')(d2)
        d2r=self._cbu(d2,'conv2d_76','conv2d_77','conv2d_78','conv2d_79')
        d2r=self._ib(d2r,'conv2d_80','conv2d_81','conv2d_82')
        s=self._relu(self._conv(d2r,self._k('conv2d_83'),self._b('conv2d_83'),'valid'))
        s=self._relu(self._conv(s,self._k('conv2d_84'),self._b('conv2d_84'),'same'))
        s=self._relu(self._conv(s,self._k('conv2d_85'),self._b('conv2d_85'),'valid'))
        d2r=self._relu(s+d2r)
        skip1=self._cbam(e1p2)
        u1=self._relu(self._conv(self._up(d2r),self._kt('conv2d_transpose_6'),self._b('conv2d_transpose_6'),'same'))
        u1=self._relu(self._conv(self._up(u1),self._kt('conv2d_transpose_7'),self._b('conv2d_transpose_7'),'same'))
        h,w=min(u1.shape[1],skip1.shape[1]),min(u1.shape[2],skip1.shape[2])
        cat1=np.concatenate([skip1[:,:h,:w,:],u1[:,:h,:w,:]],axis=-1)
        d1=self.C('conv2d_87')(cat1); d1=self.C('conv2d_88')(d1)
        d1r=self._cbu(d1,'conv2d_89','conv2d_90','conv2d_91','conv2d_92')
        d1r=self._ib(d1r,'conv2d_93','conv2d_94','conv2d_95')
        d1r=self._ib(d1r,'conv2d_96','conv2d_97','conv2d_98')
        out=self._relu(self._conv(d1r,self._k('conv2d_99'),self._b('conv2d_99'),'same'))
        return self._sig(self._conv(out,self._k('conv2d_100'),self._b('conv2d_100'),'valid'))


def load_mito(fp):
    img = tifffile.imread(fp)
    ch = int(np.argmax([img[:,:,c].mean() for c in range(img.shape[2])])) if img.ndim==3 else 0
    return (img[:,:,ch] if img.ndim==3 else img).astype(np.float32)


def predict_tiled(model, img):
    stride = TILE_SIZE - OVERLAP*2
    img_pad = np.pad(img, OVERLAP, mode='reflect')
    Hp,Wp = img_pad.shape
    psum = np.zeros((Hp,Wp), np.float32)
    pcnt = np.zeros((Hp,Wp), np.float32)
    ys = list(range(0, Hp-TILE_SIZE+1, stride))
    xs = list(range(0, Wp-TILE_SIZE+1, stride))
    if ys[-1]+TILE_SIZE < Hp: ys.append(Hp-TILE_SIZE)
    if xs[-1]+TILE_SIZE < Wp: xs.append(Wp-TILE_SIZE)
    n = len(ys)*len(xs)
    for ti,(y,x) in enumerate([(y,x) for y in ys for x in xs]):
        t = img_pad[y:y+TILE_SIZE, x:x+TILE_SIZE]
        lo,hi = np.percentile(t,[1,99.8])
        t_norm = np.clip((t-lo)/(hi-lo+1e-9),0,1)[np.newaxis,:,:,np.newaxis]
        prob = model.predict(t_norm)[0,:,:,0]
        psum[y:y+TILE_SIZE,x:x+TILE_SIZE] += prob
        pcnt[y:y+TILE_SIZE,x:x+TILE_SIZE] += 1
        print(f"    tile {ti+1}/{n}", flush=True)
    H,W = img.shape
    return psum[OVERLAP:OVERLAP+H, OVERLAP:OVERLAP+W] / np.maximum(pcnt[OVERLAP:OVERLAP+H, OVERLAP:OVERLAP+W],1)


def to_instances(prob):
    binary = (prob >= THRESHOLD).astype(np.uint8)
    dist = distance_transform_edt(binary)
    lmax = (dist == maximum_filter(dist, size=MIN_DISTANCE*2+1)) & (binary>0)
    seeds,_ = nd_label(lmax)
    from skimage.segmentation import watershed
    labels = watershed(-dist, seeds, mask=binary)
    res = np.zeros_like(labels, dtype=np.uint16)
    nid = 1
    for prop in regionprops(labels):
        if prop.area >= MIN_AREA:
            res[labels==prop.label] = nid; nid+=1
    return res


def main():
    print("Loading MoDL weights..."); model = NumpyModel(MODEL_PATH); print("OK")
    all_images = [(fp, ct) for ct,d in DATA_DIRS.items() if d.exists()
                  for fp in sorted(d.glob('*.tif'))]
    print(f"Total images: {len(all_images)}")
    stats = []
    for fp, ct in all_images:
        out_tif = OUT_PRED / fp.name
        if out_tif.exists():
            print(f"  [skip] {fp.name}"); continue
        print(f"\n[{fp.name}] ({ct})", flush=True)
        mito = load_mito(fp)
        prob = predict_tiled(model, mito)
        labeled = to_instances(prob)
        n = int(labeled.max())
        areas = [int((labeled==i).sum()) for i in range(1,n+1)]
        tifffile.imwrite(out_tif, labeled)
        # QC figure
        fig,axes = plt.subplots(1,2,figsize=(10,5))
        lo,hi = np.percentile(mito,[1,99.5])
        mn = np.clip((mito-lo)/(hi-lo+1e-9),0,1)
        axes[0].imshow(mn,cmap='gray'); axes[0].set_title('Mito'); axes[0].axis('off')
        cmap = plt.cm.get_cmap('tab20',max(n,1))
        disp = np.zeros((*labeled.shape,4))
        for iid in range(1,n+1): disp[labeled==iid]=cmap(iid%20)
        axes[1].imshow(mn,cmap='gray'); axes[1].imshow(disp,alpha=0.6)
        axes[1].set_title(f'n={n}'); axes[1].axis('off')
        plt.suptitle(f'{fp.stem} | MoDL | n={n}',fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_FIG/f'{fp.stem}.png',dpi=100,bbox_inches='tight')
        plt.close(fig)
        stats.append({'image':fp.stem,'cell_type':ct,'n_mito':n,
                      'area_median':int(np.median(areas)) if areas else 0})
        print(f"  n={n}  med_area={stats[-1]['area_median']}", flush=True)
    if stats:
        import csv as _csv
        with open(ROOT/'results'/'run34_modl_stats.csv','w',newline='') as f:
            w=_csv.DictWriter(f,fieldnames=list(stats[0].keys()))
            w.writeheader(); w.writerows(stats)
    print(f"\nDone. Predictions -> {OUT_PRED}")

if __name__ == '__main__':
    main()
