# Code for "Distortion-Free Until You Select"

Self-contained pipeline + analysis for all figures and tables.

## Pipeline (GPU; produces the four arms)

| file | role |
|---|---|
| `wm_generators_beam.py` | generators. `OpenaiGeneratorBeamStrict(subkey=False)` = strict shared key (collapse arm); `subkey=True` = **JC lift** (per-candidate domain-separated subkeys, candidate index mixed into the seed hash chain). `OpenaiGeneratorBeam` = softened baseline. |
| `pf_bon_gen.py` | candidate generation. `--wm gumbel_ss / gumbel_sk / gumbel / unwm`, `--n` prompts, `--num_cand N`. |
| `reward_only.py` | ArmoRM reward for every candidate. |
| `pf_bon_detect.py` | base-key detectors (kgw/gumbel/pf; used for softened arm + base-key nulls). |
| `pf_bon_detect_union.py` | **union detector**: `NKEYS=0` base key, `NKEYS=N` union over N subkeys with Bonferroni (`p = min(1, N·min_i p_i)`). Subkey mix must match the generator: `seed ← (seed·salt + 424242 + i) mod 2^64−1`. |
| `ppl_orig.py` | perplexity of candidates under the generating model. |
| `run_jcs.sh` | orchestrator for the main experiment (ss + sk arms: gen → reward → detect incl. union-FPR calibration). |

Environment: HF `transformers` + `torch` (one A100-40GB per arm),
`NousResearch/Meta-Llama-3.1-8B-Instruct`, `RLHFlow/ArmoRM-Llama3-8B-v0.1`,
`Dahoas/full-hh-rlhf` (test split, raw format).

Follow-up suite jobs (after `run_jcs.sh`): union detection (`NKEYS=16`) on the
unwatermarked arm's full 128 prompts (negatives for AUROC/matched-FPR), and
`ppl_orig.py` on the ss/sk arms.

## Analysis (CPU; produces the paper's figures/tables)

Run from the repo root. Real-data scripts read `$DATA`
(default `~/Desktop/iclr_wm_exp/pod_experiments/results`), which holds the
pulled outputs: `mtx16/{unwm_kgw,gumbel,unwm_gumbel}_scored.jsonl`,
`mtx16/{gumbel,unwm}_ppl.jsonl`, `jcs/{ss,sk}_{r,scored,ppl}.jsonl`,
`jcs/unwm_union_full_scored.jsonl`.

| script | output |
|---|---|
| `analysis/fig1_toy_shift.py` | Figure 1 (`figures/toy_select_shift.pdf`) — copula toy: marginal-exact, correlation-biased selection |
| `analysis/fig2_detectors.py` | Figure 2 (`figures/detector_comparison.pdf`) — union vs aggregate + closed form |
| `analysis/fig3_real_shift.py` | Figure 3 (`figures/real_shift_vs_N.pdf`) + diversity table |
| `analysis/suite_table.py` | Table 2 (full metric suite) + matched-FPR table |

## Repro map

| paper item | source |
|---|---|
| Table 1 (main) | `run_jcs.sh` arms + `analysis/suite_table.py` |
| Table 2 / matched-FPR | `analysis/suite_table.py` |
| Fig 1 | `analysis/fig1_toy_shift.py` |
| Fig 2 + Prop. 3 validation | `analysis/fig2_detectors.py` |
| Fig 3 + diversity | `analysis/fig3_real_shift.py` |
| Calibration numbers (App. B) | `run_jcs.sh` (unwm union arm) + `suite_table.py` |
