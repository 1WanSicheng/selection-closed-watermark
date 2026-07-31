import json, sys, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
DEV="cuda:0"
rm=AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(DEV).eval()
rmtok=AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",use_fast=True)
@torch.no_grad()
def reward(p,r):
    if not r or not r.strip(): return float("nan")
    try:
        ids=rmtok.apply_chat_template([{"role":"user","content":p},{"role":"assistant","content":r}],return_tensors="pt").to(DEV)
        return float(rm(ids).score.cpu().float().item())
    except Exception: return float("nan")
with open(sys.argv[1]) as f, open(sys.argv[2],"w") as g:
    for line in f:
        d=json.loads(line); p=d["prompt"]
        d["cands_reward"]=[reward(p,c) for c in d["cands"]]
        g.write(json.dumps(d)+"\n"); g.flush()
print("REWARD_DONE")
