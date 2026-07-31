import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda:0"; M="NousResearch/Meta-Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
mdl=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float16,low_cpu_mem_usage=True).to(DEV).eval()
@torch.no_grad()
def ppl(t):
    if not t or not t.strip(): return float("nan")
    e=tok(t,return_tensors="pt",truncation=True,max_length=400).to(DEV)
    if e["input_ids"].shape[1]<2: return float("nan")
    return float(torch.exp(mdl(**e,labels=e["input_ids"]).loss))
with open(sys.argv[1]) as f, open(sys.argv[2],"w") as g:
    for line in f:
        d=json.loads(line); d["cands_ppl_orig"]=[ppl(c) for c in d["cands"]]
        g.write(json.dumps({"cands_reward":d.get("cands_reward"),"cands_pvalue":d.get("cands_pvalue"),"cands_ppl_orig":d["cands_ppl_orig"]})+"\n"); g.flush()
print("PPL_DONE")
