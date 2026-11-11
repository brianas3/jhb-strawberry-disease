import sys, os, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
df = load_manifest(); X = np.load(f"{MODELS}/feats_plain.npy").astype(np.float32); y = df.class_id.values
tr, va, te = [(df.split == s).values for s in ("train","val","test")]; conf = (df.label_conflict==1).values
cls = sorted(set(y))
def mf1(a,b): return f1_score(a,b,labels=sorted(set(a)),average="macro",zero_division=0)
def fit(m,C):
    sc=StandardScaler().fit(X[m]); c=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(sc.transform(X[m]),y[m]); return sc,c
def pred(sc,c,A): return c.predict(sc.transform(A))
def fitsel(mtr, mva, Cs=(1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1,1)):
    best=None
    for C in Cs:
        sc,c=fit(mtr,C); s=mf1(y[mva],pred(sc,c,X[mva]))
        if best is None or s>best[0]: best=(s,C,sc,c)
    return best
out={}
# 1 reproduce
b=fitsel(tr&~conf,va); s,C,sc,c=b; p=pred(sc,c,X[te])
out["repro"]=dict(C=C,val=s,acc=accuracy_score(y[te],p),mf1=mf1(y[te],p))
b2=fitsel(tr,va); p2=pred(b2[2],b2[3],X[te]); out["repro_keep"]=dict(C=b2[1],val=b2[0],acc=accuracy_score(y[te],p2),mf1=mf1(y[te],p2))
# group overlap
g=df.group_id.values
out["group_overlap"]={f"{a}-{b}":len(set(g[m1])&set(g[m2])) for a,m1,b,m2 in [("tr",tr,"va",va),("tr",tr,"te",te),("va",va,"te",te)]}
# fresh phash
import imagehash
from PIL import Image
H=[]
for pth in df.fpath:
    im=Image.open(pth).convert("RGB"); H.append(imagehash.phash(im,hash_size=8).hash.flatten())
H=np.array(H,dtype=np.uint8); np.save(os.path.join(MODELS,"..","reports","redteam_phash.npy"),H)
def near(A,B,th): 
    d=(A[:,None,:]!=B[None,:,:]).sum(2); return d
d=near(H[te],H[tr],0); mn=d.min(1); out["te_vs_tr_phash_min_dist_hist"]={f"<={t}":int((mn<=t).sum()) for t in (0,4,6,10)}
idx=np.where(mn<=10)[0]; ti=np.where(te)[0]; tri=np.where(tr)[0]
rows=[]
for i in idx:
    j=tri[d[i].argmin()]; rows.append((int(ti[i]),int(j),int(mn[i]),int(y[ti[i]]),int(y[j]),int(p[i])))
nd=pd.DataFrame(rows,columns=["te","tr","dist","y_te","y_tr","pred"]); out["neardup_n"]=len(nd)
out["neardup_samelabel"]=int((nd.y_te==nd.y_tr).sum()); out["neardup_correct"]=int((nd.pred==nd.y_te).sum())
keep=np.ones(te.sum(),bool); keep[idx]=False
out["acc_excl_neardup"]=accuracy_score(y[te][keep],p[keep]); out["mf1_excl_neardup"]=mf1(y[te][keep],p[keep])
nd.to_csv("reports/redteam_neardup.csv",index=False)
# also val-vs-train
dv=near(H[va],H[tr],0).min(1); out["va_vs_tr_le10"]=int((dv<=10).sum())
# 4 conflicts per class
out["conflict_by_class"]={int(k):int(v) for k,v in pd.Series(y[conf]).value_counts().sort_index().items()}
out["train_by_class"]={int(k):int(v) for k,v in pd.Series(y[tr]).value_counts().sort_index().items()}
# multi-seed resplit: group-aware stratified 70/15/15 using group_id; test-time conflict rows excluded from val/test
res=[]
grp=g; ok=~conf
idxs=np.where(ok)[0]
for seed in range(5):
    sgk=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=seed)
    folds=[f for _,f in sgk.split(idxs,y[idxs],grp[idxs])]
    te_i=idxs[folds[0]]; va_i=idxs[folds[1]]
    mte=np.zeros(len(df),bool); mva=mte.copy(); mte[te_i]=True; mva[va_i]=True
    # groups containing conflict rows must not leak into eval: drop eval rows sharing group with any conflict row
    cg=set(grp[conf]); 
    mtr=~(mte|mva)
    # remove from train any row whose group in eval
    eg=set(grp[mte|mva]); mtr&=~np.isin(grp,list(eg))
    r={}
    for name,m in [("drop",mtr&~conf),("keep",mtr)]:
        bb=fitsel(m,mva); pp=pred(bb[2],bb[3],X[mte]); r[name]=(accuracy_score(y[mte],pp),mf1(y[mte],pp),bb[0])
    res.append(r)
for name in ("drop","keep"):
    a=np.array([r[name] for r in res]); out[f"resplit_{name}"]=dict(acc_mean=a[:,0].mean(),acc_sd=a[:,0].std(),mf1_mean=a[:,1].mean(),mf1_sd=a[:,1].std(),val_mf1_mean=a[:,2].mean(),per_seed_mf1=a[:,1].round(3).tolist())
out["resplit_diff_mf1"]=[round(r["drop"][1]-r["keep"][1],3) for r in res]
# 3 group k-fold by (class, size) cluster: leave-one-source-out proxy
w=pd.read_csv("reports/image_table.csv").set_index("path").loc[df.path]
df["w"]=w.w.values; df["h"]=w.h.values
loso={}
for cq in (3,4,11):
    m=(y==cq); print(cq,df[m].groupby(["w","h"]).size().sort_values(ascending=False).head(4).to_dict())
# for class 11: sizes vary (128..1024). Hold out largest-size-bin
# generic: hold out rows with w>=median of class for classes with spread (11): train on rest incl. all other classes
for cq in (11,):
    m=(y==cq); med=np.median(df.w[m]); print("c11 w quantiles",np.percentile(df.w[m],[0,25,50,75,100]))
    # hold out class-11 imgs w!=256 (minority resolutions)
    odd=m&(df.w.values!=256); print("c11 non-256:",odd.sum(), "of",m.sum())
    allm=tr|va|te
    ho=odd&allm; trn=allm&~ho&~conf
    sc_,c_=fit(trn,0.01) if False else fit(trn,out["repro"]["C"]); pp=pred(sc_,c_,X[ho]); loso["c11_non256_recall"]=float((pp==11).mean()); loso["c11_non256_n"]=int(ho.sum())
    # baseline: random 256-w holdout of same size
    # in-distribution recall of 256-wide test images
    m256=m&(df.w.values==256)&te; loso["c11_256_test_recall"]=float((pred(sc,c,X[m256])==11).mean())
# classes 3,4 (single res): hold out entire class -> not meaningful; instead 5-fold StratifiedGroupKFold over all non-conflict, report overall vs source cluster
# cluster-based CV: cluster=(class, width bucket); GroupKFold on clusters => class may be absent from train for single-cluster classes. Report accuracy for multi-cluster classes only.
df["cl"]=df.class_id.astype(str)+"_"+pd.cut(np.log2(df.w),[0,8.5,9.5,10.5,11.5,20]).astype(str)
allm=np.where(~conf)[0]; gkf=GroupKFold(n_splits=5); pr=np.full(len(df),-1)
for a,bidx in gkf.split(allm,y[allm],df.cl.values[allm]):
    a=allm[a]; bidx=allm[bidx]; m=np.zeros(len(df),bool); m[a]=True
    sc_,c_=fit(m,out["repro"]["C"]); pr[bidx]=pred(sc_,c_,X[bidx])
ev=allm; out["clusterCV_acc"]=accuracy_score(y[ev],pr[ev]); out["clusterCV_mf1"]=mf1(y[ev],pr[ev])
out["clusterCV_per_class_recall"]={int(k):round(float((pr[ev][y[ev]==k]==k).mean()),3) for k in cls}
# random group CV comparison
gk=GroupKFold(n_splits=5); pr2=np.full(len(df),-1)
for a,bidx in gk.split(allm,y[allm],g[allm]):
    a=allm[a]; bidx=allm[bidx]; m=np.zeros(len(df),bool); m[a]=True
    sc_,c_=fit(m,out["repro"]["C"]); pr2[bidx]=pred(sc_,c_,X[bidx])
out["groupCV_acc"]=accuracy_score(y[ev],pr2[ev]); out["groupCV_mf1"]=mf1(y[ev],pr2[ev])
out["groupCV_per_class_recall"]={int(k):round(float((pr2[ev][y[ev]==k]==k).mean()),3) for k in cls}
out["loso"]=loso
json.dump(out,open("reports/redteam_core.json","w"),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
