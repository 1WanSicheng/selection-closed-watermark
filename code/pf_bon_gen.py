import json, torch, argparse, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from wm_generators_beam import OpenaiGeneratorBeam, MarylandGeneratorBeam, PFGeneratorBeam, WmGeneratorBeam, OpenaiGeneratorBeamStrict
from datasets import load_dataset
ap=argparse.ArgumentParser()
ap.add_argument("--wm",required=True)                 # unwm|gumbel|kgw|pf
ap.add_argument("--model",default="NousResearch/Llama-2-7b-hf")
ap.add_argument("--fmt",default="raw"); ap.add_argument("--dtype",default="fp16")
ap.add_argument("--out",required=True); ap.add_argument("--n",type=int,default=128)
ap.add_argument("--num_cand",type=int,default=8); ap.add_argument("--temp",type=float,default=1.0)
ap.add_argument("--top_p",type=float,default=0.95); ap.add_argument("--gen_len",type=int,default=120)
ap.add_argument("--batch",type=int,default=2)
ap.add_argument("--delta",type=float,default=2.0)
ap.add_argument("--gamma",type=float,default=0.25)
a=ap.parse_args(); DEV="cuda:0"
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
dt=torch.bfloat16 if a.dtype=="bf16" else torch.float16
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dt,attn_implementation="sdpa",low_cpu_mem_usage=True).to(DEV).eval()
eos=model.config.eos_token_id; model.config.pad_token_id=eos[0] if isinstance(eos,(list,tuple)) else eos
ds=load_dataset("Dahoas/full-hh-rlhf",split="test")
prompts=[ds[i]["prompt"] for i in range(a.n)]; golds=[ds[i].get("chosen","") for i in range(a.n)]
def last_user(p):
    parts=re.split(r'\n\n(Human|Assistant):\s?', p); last=""; i=1
    while i+1<len(parts):
        if parts[i]=="Human": last=parts[i+1].strip()
        i+=2
    return last or p.strip()
def _tr(p):
    ids=tok.encode(p,add_special_tokens=False); return tok.decode(ids[-96:]) if len(ids)>96 else p
def fmt(p):
    if a.fmt=="chat":
        return tok.apply_chat_template([{"role":"user","content":last_user(p)}],add_generation_prompt=True,tokenize=False)
    return _tr(p)
P=[fmt(p) for p in prompts]
Q=[last_user(p) if a.fmt=="chat" else fmt(p) for p in prompts]   # question for reward
C=dict(ngram=4,seed=42,salt_key=35317)
G={"unwm":lambda:WmGeneratorBeam(model,tok,**C),
   "gumbel":lambda:OpenaiGeneratorBeam(model,tok,**C),
   "kgw":lambda:MarylandGeneratorBeam(model,tok,gamma=a.gamma,delta=a.delta,**C),
   "pf":lambda:PFGeneratorBeam(model,tok,**C),
   "gumbel_ss":lambda:OpenaiGeneratorBeamStrict(model,tok,subkey=False,**C),
   "gumbel_sk":lambda:OpenaiGeneratorBeamStrict(model,tok,subkey=True,**C)}[a.wm]()
def comp(pr,f): return (f.replace(pr,"",1).strip() or f.strip())
with open(a.out,"w") as out:
    for i in range(0,a.n,a.batch):
        bp=P[i:i+a.batch]
        cands=G.generate(bp,max_gen_len=a.gen_len,temperature=a.temp,top_p=a.top_p,num_beams=a.num_cand,num_return_sequences=a.num_cand)
        for j in range(len(bp)):
            out.write(json.dumps({"prompt":Q[i+j],"gold":golds[i+j],"cands":[comp(bp[j],c) for c in cands[j]]})+"\n"); out.flush()
        print("done",i+len(bp),flush=True)
print("BON_GEN_DONE",flush=True)
