#!/bin/bash
cd /data/wansicheng2/alignmark
export HF_HOME=/data/wansicheng2/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/data/wansicheng2/env/bin/python
M=NousResearch/Meta-Llama-3.1-8B-Instruct; V=128256
N=128; NC=16; GL=200; D=jcs; mkdir -p $D; LOG=$D/log; : > $LOG
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "JCS: strict shared vs strict subkey (N=16cand x $N prompts)"
CUDA_VISIBLE_DEVICES=4 $PY -B pf_bon_gen.py --wm gumbel_ss --model $M --fmt raw --out $D/ss.jsonl --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_ss.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY -B pf_bon_gen.py --wm gumbel_sk --model $M --fmt raw --out $D/sk.jsonl --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_sk.log 2>&1 &
wait
say "GEN done: ss=$(wc -l <$D/ss.jsonl) sk=$(wc -l <$D/sk.jsonl)"
CUDA_VISIBLE_DEVICES=4 $PY -B reward_only.py $D/ss.jsonl $D/ss_r.jsonl > $D/rew_ss.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY -B reward_only.py $D/sk.jsonl $D/sk_r.jsonl > $D/rew_sk.log 2>&1 &
wait
say "REWARD done"
head -32 mtx16/unwm.jsonl > $D/unwm32.jsonl
CUDA_VISIBLE_DEVICES=4 $PY -B pf_bon_detect_union.py $D/ss_r.jsonl $D/ss_scored.jsonl $V $M 0  > $D/det_ss.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY -B pf_bon_detect_union.py $D/sk_r.jsonl $D/sk_scored.jsonl $V $M 16 > $D/det_sk_union.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -B pf_bon_detect_union.py $D/unwm32.jsonl $D/unwm_union_scored.jsonl $V $M 16 > $D/det_unwm_union.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY -B pf_bon_detect_union.py $D/sk_r.jsonl $D/sk_base_scored.jsonl $V $M 0 > $D/det_sk_base.log 2>&1 &
wait
say "DETECT done: ss=$(wc -l <$D/ss_scored.jsonl) sk_union=$(wc -l <$D/sk_scored.jsonl) unwm_union=$(wc -l <$D/unwm_union_scored.jsonl) sk_base=$(wc -l <$D/sk_base_scored.jsonl)"
say "ALL DONE"
