import json, sys, torch, numpy as np
from scipy import special
from transformers import AutoTokenizer
# argv: in out V MODEL NKEYS   (NKEYS=0 -> base key only; NKEYS=k -> union over k subkeys)
V=int(sys.argv[3]); MODEL=sys.argv[4]; NKEYS=int(sys.argv[5]); DEV="cuda:0"
tok=AutoTokenizer.from_pretrained(MODEL)
NG=4;SEED=42;SALT=35317;MIX=424242; rng=torch.Generator(device=DEV)
def sd(ng):
    s=SEED
    for i in ng: s=(s*SALT+int(i))%(2**64-1)
    return s
@torch.no_grad()
def pvalue_key(ids,cid):
    n=0; sc=0.0
    for pos in range(NG+1,len(ids)):
        base=sd(ids[pos-NG:pos])
        seed=base if cid is None else (base*SALT+MIX+cid)%(2**64-1)
        rng.manual_seed(seed)
        rs=torch.rand(V,generator=rng,device=DEV); r=float(rs[ids[pos]])
        sc += -np.log(max(1-r,1e-9)); n+=1
    if n==0: return 1.0
    return float(max(special.gammaincc(n,sc),1e-300))
with open(sys.argv[1]) as f, open(sys.argv[2],"w") as g:
    for line in f:
        d=json.loads(line); pv=[]
        for c in d["cands"]:
            ids=tok.encode(c,add_special_tokens=False)
            if len(ids)<NG+2: pv.append(1.0); continue
            if NKEYS==0: pv.append(pvalue_key(ids,None))
            else:
                ps=[pvalue_key(ids,cid) for cid in range(NKEYS)]
                pv.append(float(min(1.0,NKEYS*min(ps))))   # union + Bonferroni
        d["cands_pvalue"]=pv; g.write(json.dumps(d)+"\n"); g.flush()
print("DETECT_UNION_DONE")
