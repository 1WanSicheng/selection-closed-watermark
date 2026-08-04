#!/bin/bash
cd /data/wansicheng2/alignmark
export HF_HOME=/data/wansicheng2/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/data/wansicheng2/env/bin/python
M=NousResearch/Meta-Llama-3.1-8B; V=128256
N=128; NC=16; GL=200; D=matrix_base; LOG=$D/log2; : > $LOG
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "SUPP: shared-key gumbel_ss + pf_ss on base"
CUDA_VISIBLE_DEVICES=5 $PY -B pf_bon_gen.py --wm gumbel_ss --model $M --fmt raw --out $D/gss.jsonl  --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_gss.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -B pf_bon_gen.py --wm pf_ss     --model $M --fmt raw --out $D/pfss.jsonl --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_pfss.log 2>&1 &
wait
say "SUPP GEN done: gss=$(wc -l <$D/gss.jsonl) pfss=$(wc -l <$D/pfss.jsonl)"
CUDA_VISIBLE_DEVICES=5 $PY -B reward_only.py $D/gss.jsonl  $D/gss_r.jsonl  > $D/rew_gss.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -B reward_only.py $D/pfss.jsonl $D/pfss_r.jsonl > $D/rew_pfss.log 2>&1 &
wait
say "SUPP REWARD done"
CUDA_VISIBLE_DEVICES=5 $PY -B pf_bon_detect.py $D/gss_r.jsonl  $D/gss_scored.jsonl  gumbel $V $M > $D/det_gss.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -B pf_bon_detect.py $D/pfss_r.jsonl $D/pfss_scored.jsonl pf     $V $M > $D/det_pfss.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY -B pf_bon_detect.py $D/unwm.jsonl   $D/null_pf.jsonl     pf     $V $M > $D/det_nullpf.log 2>&1 &
wait
say "SUPP ALL DONE: gss=$(wc -l <$D/gss_scored.jsonl) pfss=$(wc -l <$D/pfss_scored.jsonl)"
