import sys, os, io, numpy as np, torch, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import joblib
from PIL import Image, ImageOps
from torchvision import transforms as T
from transformers import AutoModel
from sklearn.metrics import f1_score, accuracy_score
torch.set_num_threads(8)
df=load_manifest(); te=(df.split=="test").values; d=df[te].reset_index(drop=True); y=d.class_id.values
pb=joblib.load(f"{MODELS}/probe_best.joblib"); sc,clf=pb["scaler"],pb["clf"]
m=AutoModel.from_pretrained("facebook/dinov2-small").eval()
tf=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def jpeg(im,q=30):
    b=io.BytesIO(); im.save(b,"JPEG",quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def lowres(im):  # destroy resolution/sharpness cues: shrink to 64px short side then back, then fixed 256 square
    w,h=im.size; s=64/min(w,h); im2=im.resize((max(8,int(w*s)),max(8,int(h*s))),Image.BILINEAR); return im2.resize((256,256),Image.BICUBIC)
def fixed(im): return im.resize((256,256),Image.BICUBIC)  # fixed-size squash (no aspect/res info)
def gray(im): return ImageOps.grayscale(im).convert("RGB")
def mask(im):
    w,h=im.size; return im.crop((int(.15*w),int(.15*h),int(.85*w),int(.85*h)))
V={"identity":lambda im:im,"fixed256":fixed,"lowres64":lowres,"jpeg_q30":jpeg,"grayscale":gray,"crop_outer15":mask}
def feats(fn):
    out=[]
    for i in range(0,len(d),32):
        x=torch.stack([tf(fn(Image.open(p).convert("RGB"))) for p in d.fpath[i:i+32]])
        with torch.no_grad(): h=m(pixel_values=x).last_hidden_state
        out.append(torch.cat([h[:,0],h[:,1:].mean(1)],1).numpy())
    return np.concatenate(out)
def mf1(a,b): return f1_score(a,b,labels=sorted(set(a)),average="macro",zero_division=0)
res={}
for k,fn in V.items():
    F=feats(fn); p=clf.predict(sc.transform(F)); r=dict(acc=accuracy_score(y,p),mf1=mf1(y,p))
    for c in (3,4,11): r[f"rec{c}"]=float((p[y==c]==c).mean())
    r["pred_dist"]={int(a):int(b) for a,b in zip(*np.unique(p,return_counts=True))}
    if k=="identity": r["agree_with_cached"]=float((p==clf.predict(sc.transform(np.load(f'{MODELS}/feats_plain.npy').astype(np.float32)[te]))).mean())
    res[k]=r; print(k,{a:b for a,b in r.items() if a!="pred_dist"},flush=True)
json.dump(res,open("reports/redteam_stress.json","w"),indent=1)
