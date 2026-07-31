import json, sys, torch, numpy as np
from scipy import special
from transformers import AutoTokenizer
DET=sys.argv[3]; V=int(sys.argv[4]); MODEL=sys.argv[5]; GAMMA=float(sys.argv[6]) if len(sys.argv)>6 else 0.25; DEV="cuda:0"
tok=AutoTokenizer.from_pretrained(MODEL)
NG=4;SEED=42;SALT=35317; rng=torch.Generator(device=DEV)
def sd(ng):
    s=SEED
    for i in ng: s=(s*SALT+int(i))%(2**64-1)
    return s
@torch.no_grad()
def pvalue(ids):
    n=0; sc=0.0; green=0
    for pos in range(NG+1,len(ids)):
        rng.manual_seed(sd(ids[pos-NG:pos]))
        if DET=="kgw":
            perm=torch.randperm(V,generator=rng,device=DEV); gm=torch.zeros(V,dtype=torch.bool,device=DEV); gm[perm[:int(GAMMA*V)]]=True
            green+=int(gm[ids[pos]].item()); n+=1
        else:
            rs=torch.rand(V,generator=rng,device=DEV); r=float(rs[ids[pos]])
            sc += -np.log(max(1-r,1e-9)) if DET=="gumbel" else -np.log(max(r,1e-9)); n+=1
    if n==0: return 1.0
    if DET=="kgw": z=(green-GAMMA*n)/np.sqrt(GAMMA*(1-GAMMA)*n); return float(max(0.5*special.erfc(z/np.sqrt(2)),1e-300))
    return float(max(special.gammaincc(n,sc),1e-300))
with open(sys.argv[1]) as f, open(sys.argv[2],"w") as g:
    for line in f:
        d=json.loads(line); pv=[]
        for c in d["cands"]:
            ids=tok.encode(c,add_special_tokens=False); pv.append(pvalue(ids) if len(ids)>=NG+2 else 1.0)
        d["cands_pvalue"]=pv; g.write(json.dumps(d)+"\n"); g.flush()
print("DETECT_DONE")
