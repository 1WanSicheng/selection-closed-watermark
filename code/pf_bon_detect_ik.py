import json, sys, torch, numpy as np
from scipy import special
from transformers import AutoTokenizer
# argv: in out V MODEL   -- union detector over the 16 stored independent salt keys
V=int(sys.argv[3]); MODEL=sys.argv[4]; DEV="cuda:0"
IK_SALTS=[2711935942143306561, 3542190141494331841, 5411067959362706679, 8777113617077343983, 4280597930738562321, 5776454549905894741, 8652403975426539789, 2086892414829495983, 581424696065941315, 2400990565321247165, 220881029758427831, 2432474093746260921, 5559555676424321225, 8151898350809513727, 6943724312856700727, 4665630377395291769]
tok=AutoTokenizer.from_pretrained(MODEL)
NG=4;SEED=42; rng=torch.Generator(device=DEV)
def sd(ng,salt):
    s=SEED
    for i in ng: s=(s*salt+int(i))%(2**64-1)
    return s
@torch.no_grad()
def pvalue_salt(ids,salt):
    n=0; sc=0.0
    for pos in range(NG+1,len(ids)):
        rng.manual_seed(sd(ids[pos-NG:pos],salt))
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
            ps=[pvalue_salt(ids,s) for s in IK_SALTS]
            pv.append(float(min(1.0,len(IK_SALTS)*min(ps))))
        d["cands_pvalue"]=pv; g.write(json.dumps(d)+chr(10)); g.flush()
print("DETECT_IK_DONE")
