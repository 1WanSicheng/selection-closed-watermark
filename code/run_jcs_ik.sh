#!/bin/bash
cd /data/wansicheng2/alignmark
export HF_HOME=/data/wansicheng2/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/data/wansicheng2/env/bin/python
M=NousResearch/Meta-Llama-3.1-8B-Instruct; V=128256
N=128; NC=16; GL=200; D=jcs_ik; mkdir -p $D; LOG=$D/log; : > $LOG
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "IK oracle: one independently drawn key per candidate (16cand x $N prompts)"
CUDA_VISIBLE_DEVICES=0 $PY -B pf_bon_gen.py --wm gumbel_ik --model $M --fmt raw --out $D/ik.jsonl --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_ik.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -B pf_bon_detect_ik.py mtx16/unwm.jsonl $D/unwm_ik_scored.jsonl $V $M > $D/det_null.log 2>&1 &
wait
say "GEN done: ik=$(wc -l <$D/ik.jsonl) null=$(wc -l <$D/unwm_ik_scored.jsonl)"
CUDA_VISIBLE_DEVICES=0 $PY -B reward_only.py $D/ik.jsonl $D/ik_r.jsonl > $D/rew_ik.log 2>&1
say "REWARD done: $(wc -l <$D/ik_r.jsonl)"
CUDA_VISIBLE_DEVICES=0 $PY -B pf_bon_detect_ik.py $D/ik_r.jsonl $D/ik_scored.jsonl $V $M > $D/det_ik.log 2>&1
say "DETECT done: $(wc -l <$D/ik_scored.jsonl)"
say "ALL DONE"
