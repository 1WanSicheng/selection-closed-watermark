import json, torch, argparse, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from wm_generators_beam import (WmGeneratorBeam, OpenaiGeneratorBeam,
    MarylandGeneratorBeam, OpenaiGeneratorBeamStrict)

ap = argparse.ArgumentParser()
ap.add_argument("--wm", required=True)   # unwm|gumbel|kgw|gumbel_ss|gumbel_sk
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--num_cand", type=int, default=16)
ap.add_argument("--gen_len", type=int, default=288)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--delta", type=float, default=2.0)
ap.add_argument("--gamma", type=float, default=0.25)
ap.add_argument("--log_ids", action="store_true")
a = ap.parse_args(); DEV = "cuda:0"

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(DEV).eval()
eos = model.config.eos_token_id
model.config.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos

def _rd(p):
    return [json.loads(l) for l in open(p)]
train, test = _rd("gsm/gsm8k_train.jsonl"), _rd("gsm/gsm8k_test.jsonl")

def clean(ans):
    return re.sub(r"<<[^>]*>>", "", ans)

# 4 fixed short exemplars from train (deterministic scan)
shots = []
for ex in train:
    if len(ex["question"]) + len(ex["answer"]) < 420:
        shots.append("Question: %s\nAnswer: %s" % (ex["question"], clean(ex["answer"])))
        if len(shots) == 4: break
PREFIX = "\n\n".join(shots) + "\n\n"

def gold_of(ans):
    return ans.split("####")[-1].strip().replace(",", "").replace("$", "")

C = dict(ngram=4, seed=42, salt_key=35317)
G = {"unwm":      lambda: WmGeneratorBeam(model, tok, **C),
     "gumbel":    lambda: OpenaiGeneratorBeam(model, tok, **C),
     "kgw":       lambda: MarylandGeneratorBeam(model, tok, gamma=a.gamma, delta=a.delta, **C),
     "gumbel_ss": lambda: OpenaiGeneratorBeamStrict(model, tok, subkey=False, **C),
     "gumbel_sk": lambda: OpenaiGeneratorBeamStrict(model, tok, subkey=True, log_tau=a.log_ids, **C)}[a.wm]()

def trunc(pr, full):
    t = full.replace(pr, "", 1)
    cut = t.find("\nQuestion:")
    if cut > 0: t = t[:cut]
    return t.strip()

with open(a.out, "w") as out:
    for i in range(a.n):
        q = test[i]["question"]
        prompt = PREFIX + "Question: %s\nAnswer:" % q
        ptok = tok.encode(prompt, add_special_tokens=False)
        assert len(ptok) < 690, "prompt too long: %d" % len(ptok)
        cands = G.generate([prompt], max_gen_len=a.gen_len, temperature=a.temp,
                           top_p=a.top_p, num_beams=a.num_cand,
                           num_return_sequences=a.num_cand)
        rec = {"question": q, "gold": gold_of(test[i]["answer"]), "prompt": prompt,
               "cands": [trunc(prompt, c) for c in cands[0]]}
        if a.log_ids and getattr(G, "tau_log", None):
            eos_ids = model.config.eos_token_id
            eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
            rows = []
            for log in G.tau_log:
                ids = []
                for t_id, p, tau in log:
                    if t_id in eos_set: break
                    ids.append(t_id)
                rows.append(ids)
            rec["row_ids"] = rows
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print("done", i + 1, flush=True)
print("GSM_GEN_DONE", flush=True)
