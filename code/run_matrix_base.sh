#!/bin/bash
cd /data/wansicheng2/alignmark
export HF_HOME=/data/wansicheng2/hf_cache TOKENIZERS_PARALLELISM=false
PY=/data/wansicheng2/env/bin/python
M=NousResearch/Meta-Llama-3.1-8B; V=128256
N=128; NC=16; GL=200; D=matrix_base; mkdir -p $D; LOG=$D/log; : > $LOG
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "STAGE0 download base model"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 $PY - <<'PYEOF' >> $LOG 2>&1
from huggingface_hub import snapshot_download
snapshot_download("NousResearch/Meta-Llama-3.1-8B", allow_patterns=["*.json","*.safetensors","tokenizer*","*.model"])
print("DOWNLOAD_DONE")
PYEOF
say "STAGE0 done"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
say "STAGE1 generation x5"
CUDA_VISIBLE_DEVICES=0 $PY -B pf_bon_gen.py --wm unwm      --model $M --fmt raw --out $D/unwm.jsonl  --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_unwm.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -B pf_bon_gen.py --wm kgw       --model $M --fmt raw --out $D/kgw.jsonl   --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 --delta 2.0 --gamma 0.25 > $D/gen_kgw.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -B pf_bon_gen.py --wm gumbel    --model $M --fmt raw --out $D/gsoft.jsonl --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_gsoft.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -B pf_bon_gen.py --wm gumbel_sk --model $M --fmt raw --out $D/gsk.jsonl   --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_gsk.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY -B pf_bon_gen.py --wm pf_sk     --model $M --fmt raw --out $D/pfsk.jsonl  --n $N --num_cand $NC --gen_len $GL --batch 1 --temp 1.0 --top_p 0.95 > $D/gen_pfsk.log 2>&1 &
wait
say "STAGE1 done: $(wc -l <$D/unwm.jsonl) $(wc -l <$D/kgw.jsonl) $(wc -l <$D/gsoft.jsonl) $(wc -l <$D/gsk.jsonl) $(wc -l <$D/pfsk.jsonl)"

say "STAGE2 rewards x5"
CUDA_VISIBLE_DEVICES=0 $PY -B reward_only.py $D/unwm.jsonl  $D/unwm_r.jsonl  > $D/rew_unwm.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -B reward_only.py $D/kgw.jsonl   $D/kgw_r.jsonl   > $D/rew_kgw.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -B reward_only.py $D/gsoft.jsonl $D/gsoft_r.jsonl > $D/rew_gsoft.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -B reward_only.py $D/gsk.jsonl   $D/gsk_r.jsonl   > $D/rew_gsk.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY -B reward_only.py $D/pfsk.jsonl  $D/pfsk_r.jsonl  > $D/rew_pfsk.log 2>&1 &
wait
say "STAGE2 done"

say "STAGE3 detection x8"
CUDA_VISIBLE_DEVICES=0 $PY -B pf_bon_detect.py $D/gsoft_r.jsonl $D/gsoft_scored.jsonl gumbel $V $M > $D/det_gsoft.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -B pf_bon_detect.py $D/kgw_r.jsonl   $D/kgw_scored.jsonl   kgw    $V $M 0.25 > $D/det_kgw.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -B pf_bon_detect_union.py $D/gsk_r.jsonl  $D/gsk_scored.jsonl  $V $M 16    > $D/det_gsk.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -B pf_bon_detect_union.py $D/pfsk_r.jsonl $D/pfsk_scored.jsonl $V $M 16 pf > $D/det_pfsk.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY -B pf_bon_detect.py $D/unwm.jsonl $D/null_g.jsonl    gumbel $V $M > $D/det_nullg.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY -B pf_bon_detect.py $D/unwm.jsonl $D/null_kgw.jsonl  kgw    $V $M 0.25 > $D/det_nullk.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -B pf_bon_detect_union.py $D/unwm.jsonl $D/null_gu.jsonl  $V $M 16    > $D/det_nullgu.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY -B pf_bon_detect_union.py $D/unwm.jsonl $D/null_pfu.jsonl $V $M 16 pf > $D/det_nullpfu.log 2>&1 &
wait
say "STAGE3 done: gsoft=$(wc -l <$D/gsoft_scored.jsonl) kgw=$(wc -l <$D/kgw_scored.jsonl) gsk=$(wc -l <$D/gsk_scored.jsonl) pfsk=$(wc -l <$D/pfsk_scored.jsonl)"
say "ALL DONE"
