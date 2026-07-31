# Revision Plan: Distortion-Free Watermarking under Alignment and Test-Time Scaling

## 0. Revision Goal

The paper should not be framed as a deployment paper or as a paper about implementation hygiene. Its central claim should be:

> **Distortion-free watermarking is usually analyzed at the level of a single generation, whereas alignment and test-time scaling operate on a set of candidates through selection. Once selection is introduced, the relevant object is no longer the marginal law of one watermarked sample, but the joint law of the candidate pool.**

The paper is therefore about the compatibility between:

- distortion-free watermarking;
- inference-time alignment;
- Best-of-\(N\), reranking, verification, rejection sampling, and self-consistency;
- test-time compute scaling.

The contribution is not primarily “use different seeds.” The contribution is to identify that the standard distortion-free guarantee is insufficient for selection-based inference, formalize the missing guarantee, characterize it, and provide the minimal construction that restores it.

---

# 1. Recommended Core Story

## 1.1 Main tension

A distortion-free watermark certifies that one watermarked generation has the same distribution as one unwatermarked generation:

\[
Y \sim Q_x.
\]

However, alignment and test-time scaling commonly generate a candidate pool

\[
Y_1,\ldots,Y_N
\]

and then apply a selector

\[
S(Y_1,\ldots,Y_N)
\]

based on a reward model, verifier, voting rule, rejection criterion, or another inference-time objective.

The selector acts on the **joint candidate distribution**:

\[
P_{Y_{1:N}},
\]

not merely on the \(N\) marginals \(P_{Y_i}\).

Therefore,

\[
P_{Y_i}=Q_x \quad \forall i
\]

does not imply that selection behaves as it would on \(N\) independent unwatermarked samples.

The paper should present this as a mismatch between:

\[
\text{single-sample watermark correctness}
\]

and

\[
\text{selection-based test-time scaling}.
\]

## 1.2 Flagship failure

For deterministic context-hashed schemes with one shared key-derived stream, all candidates generated from the same prompt can remain identical. Then Best-of-\(N\) has no diversity to exploit and loses its entire alignment gain.

This is the sharpest motivating example, but it should not be presented as the whole scientific problem.

The more general point is:

> Even when candidates are not identical and every candidate marginal is exactly correct, dependence among candidates can change the output distribution induced by selection and attenuate test-time scaling gains.

Thus, complete collapse is the extreme case of a broader joint-distribution failure.

## 1.3 Correct scientific object

Define \(N\)-selection-closed distortion-freeness by requiring the candidate tuple to match \(N\) independent samples from the reference model:

\[
P_{Y_{1:N}}=Q_x^{\otimes N},
\]

or to be computationally indistinguishable from it in the single-master-key construction.

This definition directly targets inference-time alignment and test-time scaling because every key-independent selector is a function of the candidate tuple.

## 1.4 Minimal repair

Once the correct target is identified, the repair is deliberately simple:

- use independently sampled candidate keys for exact joint closure; or
- derive candidate-specific subkeys from one master key through domain separation for computational joint closure.

The simplicity should be presented as a strength:

> The failure is structural, but the correction is minimal once the correct guarantee is stated.

---

# 2. Reposition the Paper

## 2.1 What the paper is about

The paper is about:

- watermarking under Best-of-\(N\);
- watermarking under inference-time alignment;
- watermarking under reward-model selection;
- watermarking under verifier-guided inference;
- watermarking under self-consistency and reranking;
- the interaction between watermark guarantees and test-time compute scaling.

## 2.2 What the paper is not about

Avoid presenting it as primarily about:

- deployment correctness;
- production systems;
- an implementation bug;
- seed management;
- a new watermarking algorithm;
- a new alignment algorithm.

Recommended sentence:

> We do not propose a new watermarking decoder or a new inference-time alignment rule. We identify the distributional condition under which an existing watermarking scheme composes correctly with candidate selection, and give a minimal lift that enforces this condition.

## 2.3 Recommended one-sentence pitch

> Existing distortion-free watermarks preserve one sample at a time, but test-time alignment selects from many samples; we show that this mismatch can eliminate the entire Best-of-\(N\) gain, characterize the required joint guarantee, and restore it with candidate-separated watermark randomness.

Alternative, more theory-oriented:

> Marginal distortion-freeness is insufficient for selection-based inference; joint preservation is the necessary and sufficient condition for watermarking to compose with Best-of-\(N\) and other test-time selectors.

---

# 3. Revision of the Title and Terminology

## 3.1 Title

The existing title is strong and memorable:

> **Distortion-Free Until You Select: Watermarking Under Best-of-\(N\) Inference**

It can be retained.

Possible alternatives:

> **Distortion-Free Until You Select: Watermarking under Test-Time Scaling**

> **When Distortion-Free Watermarking Meets Best-of-\(N\)**

> **Jointly Distortion-Free Watermarking for Selection-Based Inference**

The current title is probably the best because it communicates the failure immediately.

## 3.2 Avoid “deployment” terminology

Replace:

- deployment-level guarantee;
- deployed output;
- deployed system;
- deployed scheme family;
- production pipeline.

With:

- selection-level guarantee;
- post-selection output;
- inference-time pipeline;
- test-time scaling pipeline;
- selection-based inference;
- watermarking under inference-time alignment.

Example:

**Before**

> Existing distortion-free watermarks certify the wrong deployment object.

**After**

> Existing distortion-free watermarks certify the marginal law of one generation, while inference-time alignment operates on the joint law of a candidate set.

## 3.3 Rename the construction if necessary

“JCN” or “jointly-coupled” may sound as though the method introduces more coupling.

Possible alternatives:

- Candidate-Separated Watermarking;
- Selection-Closed Lift;
- Joint-Preserving Lift;
- Candidate-Indexed Keying;
- Selection-Compatible Watermarking;
- Jointly Independent Candidate lift.

A conservative choice is:

> **Selection-Closed Lift (SCL)**

Then describe the mechanism as candidate-indexed domain separation.

If the current name is already used extensively, it can remain, but the paper should clarify:

> “Jointly coupled” refers to controlling the full joint law, not to increasing candidate dependence.

---

# 4. Rewrite the Abstract

## Recommended Abstract

Distortion-free language-model watermarks guarantee that a single watermarked generation has the same marginal law as unwatermarked text. Inference-time alignment and test-time scaling, however, commonly generate multiple candidates and select among them using a reward model, verifier, reranker, rejection rule, or self-consistency procedure. We show that the standard marginal guarantee is not preserved by such selection. In deterministic context-hashed watermarking schemes, reusing one key-derived randomness stream across candidates can make all \(N\) candidates identical, eliminating the entire Best-of-\(N\) alignment gain. More generally, candidate dependence can bias the selected output even when every candidate marginal is exactly correct.

We introduce \(N\)-selection-closed distortion-freeness, which requires the candidate tuple to match \(N\) independent unwatermarked samples. We prove that joint preservation is necessary and sufficient for closure under every key-independent selector, whereas marginal preservation alone provides no control over the post-selection distribution.

We then give a minimal, scheme-agnostic lift based on candidate-specific subkeys. Independently sampled subkeys yield exact joint closure, while PRF-derived subkeys yield computational joint closure under a single master key. Existing detectors lift through a union test, with an \(O(\sqrt{\log N})\) statistical threshold penalty and \(N\)-fold parallel detector evaluation.

On LLaMA-3.1-8B-Instruct, shared-key watermarking produces only \(1/16\) unique candidates and forfeits the full Best-of-\(16\) reward gain. Candidate-separated subkeys restore \(16/16\) unique candidates, recover essentially all available Best-of-\(N\) gain, and retain strong post-selection detection. We reproduce the same failure and correction across a second distortion-free decoder family and multiple inference-time selectors.

---

# 5. Rewrite the Introduction

## 5.1 Opening paragraph

Start from the recent use of test-time scaling:

> Test-time alignment increasingly improves language-model outputs by generating multiple candidates and selecting among them. Best-of-\(N\) uses a reward model, verifier-guided inference selects a verified solution, rejection sampling filters low-quality outputs, and self-consistency aggregates multiple reasoning traces. These methods obtain their gain from candidate diversity and from the joint structure of the candidate pool.

Then introduce watermarking:

> Distortion-free watermarking, by contrast, is usually certified one generation at a time: after averaging over the secret randomness, a single watermarked sample follows the same law as the unwatermarked model.

Then state the mismatch:

> These two views do not automatically compose. A marginal guarantee for each candidate does not determine the distribution produced after a selector observes all candidates.

This immediately anchors the paper in alignment and test-time scaling rather than deployment.

## 5.2 Central question

Use:

> **Does distortion-free watermarking remain distortion-free after inference-time selection?**

or:

> **When does a distortion-free watermark compose correctly with Best-of-\(N\) and other test-time selectors?**

The second question is theoretically stronger.

## 5.3 Explain the collapse

State the deterministic shared-key failure clearly:

> Under a deterministic context-hashed watermark, the token at each step is fixed by the prompt, the key, and the preceding context. If all candidates use the same key-derived stream, they choose the same first token. Their contexts then remain identical, so the candidates remain identical by induction. Best-of-\(N\) is reduced to Best-of-1.

Then immediately generalize:

> This complete collapse is only the endpoint of the problem. Partial candidate correlation also reduces effective test-time compute: a selector receives fewer independent opportunities to find a high-reward output, and the selected distribution can differ from the independent unwatermarked reference even though each candidate marginal is exact.

## 5.4 Add a direct objection-handling paragraph

### Why this is not merely a seeding issue

> Assigning different randomness to different candidates is an operationally natural fix, but it does not answer the underlying question: what property must a watermarked candidate pool satisfy so that an arbitrary inference-time selector behaves as it would on unwatermarked samples? The issue persists beyond complete seed reuse. Any dependence among candidates can alter the output of a reward-based selector while leaving every marginal unchanged. Our contribution is to identify and characterize the required joint guarantee; candidate-specific domain separation is then the minimal construction that realizes it.

This paragraph should appear in the first two pages.

## 5.5 Replace “Folklore vs. what is new”

The existing framing is defensive. Rename it:

### From candidate diversity to selection correctness

Suggested content:

> It is well known that deterministic decoding with reused randomness can reduce candidate diversity. Our focus is different: we ask whether the distributional guarantee of a distortion-free watermark survives a downstream selector. This requires reasoning about the joint candidate law, not only whether candidates happen to be distinct. We show that (i) exact marginals can still yield a biased post-selection distribution, (ii) joint preservation is exactly the condition needed for selector-universal closure, and (iii) candidate-separated randomness restores the reference behavior for Best-of-\(N\), reranking, rejection sampling, and other key-independent selectors.

---

# 6. Rewrite the Contributions

## Recommended Contributions Section

1. **A failure of marginal distortion-freeness under test-time selection.**  
   We show that single-sample distortion-freeness does not imply correctness after Best-of-\(N\), reward-model reranking, verifier selection, rejection sampling, or self-consistency. Under shared deterministic watermark randomness, all candidates can collapse to one sample, eliminating the entire test-time alignment gain.

2. **The correct joint guarantee.**  
   We define \(N\)-selection-closed distortion-freeness and prove that preservation of the candidate joint law is necessary and sufficient for closure under every key-independent selector. Marginal preservation alone gives no control over the selected output.

3. **A minimal selection-compatible lift.**  
   We show that independently keyed candidates provide exact joint closure, and that candidate-specific subkeys derived from one master key provide computational joint closure under PRF security. The construction applies to existing keyed watermarking decoders without modifying their token-level sampling rule.

4. **Detection after candidate separation.**  
   We lift the base detector through a union test over candidate subkeys. The statistical penalty is \(O(\sqrt{\log N})\) in the detection threshold, while the computational cost consists of \(N\) parallel base-detector evaluations.

5. **Validation across alignment selectors and watermark families.**  
   Across hard and soft Best-of-\(N\), rejection sampling, filter-and-sample, and other selectors, candidate-separated watermarking restores the reward and diversity behavior of independent inference while retaining strong watermark detection. The same pattern holds for two deterministic distortion-free decoder families.

---

# 7. Fix the Exact-versus-Computational Guarantee

This is the highest-priority soundness revision.

## 7.1 Separate three levels

### Marginal distortion-freeness

For each candidate,

\[
Y_i \sim Q_x.
\]

This is the original single-sample property.

### Exact joint closure

With independently sampled candidate keys \(K_1,\ldots,K_N\),

\[
(Y_1,\ldots,Y_N)\sim Q_x^{\otimes N}.
\]

This gives exact selection closure.

### Computational joint closure

With

\[
K_i=\operatorname{PRF}(K_{\mathrm{master}},i),
\]

the candidate tuple is computationally indistinguishable from \(Q_x^{\otimes N}\).

This gives computational closure for efficient selectors and distinguishers.

## 7.2 Terminology table

| Construction | Marginal law | Candidate joint law | Selection guarantee |
|---|---|---|---|
| Shared deterministic key | Exact marginal | Correlated or degenerate | Not selection-closed |
| Independent candidate keys | Exact marginal | Exact product law | Exact closure |
| PRF-derived candidate subkeys | Exact marginal | Computationally product-like | Computational closure |
| Softened shared-key sampler | Generally altered | Partially decorrelated | No exact marginal or joint guarantee |

Use this table in the main paper or appendix.

## 7.3 Replace unsafe claims

Replace:

> remains strictly distortion-free

with:

> retains the base scheme’s exact marginal law and achieves computational selection closure.

Replace:

> exact closure under one master key

with:

> computational closure under one master key.

Replace:

> strict subkey

with:

> deterministic subkey decoder

or:

> candidate-separated subkey construction.

“Strict” can still describe the base argmax decoder, but should not be used as a synonym for exact joint closure.

---

# 8. Remove the Incorrect Impossibility Claim

Delete:

> Exact distortion-freeness and detectability are jointly unattainable.

It is not needed and invites a direct counterexample.

## Replacement Remark

> **Remark (Exact versus compact keying).** Exact joint closure can be obtained by independently sampling and storing one secret key per candidate. The single-master-key construction replaces this \(O(N)\) key material with deterministic subkey derivation. Its guarantee is computational rather than information-theoretic, with the gap controlled by the PRF distinguishing advantage.

This makes the construction’s value precise:

- exact independent keys are the oracle;
- master-key derivation is the compact realization;
- no unsupported impossibility statement is required.

---

# 9. Split the Lifting Theorem

The current theorem should be split for clarity.

## Theorem 2: Exact Independent-Key Lift

Let \(W_K\) be a keyed sampler whose key-averaged output distribution is \(\mu_W\). Draw

\[
K_1,\ldots,K_N \overset{\mathrm{i.i.d.}}{\sim}\mathcal K
\]

and generate

\[
Y_i\sim W_{K_i}(x).
\]

Then

\[
(Y_1,\ldots,Y_N)\sim \mu_W^{\otimes N}.
\]

If \(\mu_W=Q_x\), the construction is exactly \(N\)-selection-closed.

## Theorem 3: Single-Master-Key Realization

Let

\[
K_i=\operatorname{PRF}(K_{\mathrm{master}},\texttt{candidate}=i).
\]

Under PRF security, the resulting tuple is computationally indistinguishable from \(\mu_W^{\otimes N}\). Therefore, every efficient key-independent selector produces an output computationally indistinguishable from the same selector applied to independent samples from \(\mu_W\).

This split prevents exact and computational statements from being mixed.

---

# 10. Refine the Definition and Characterization

## 10.1 Definition

Recommended definition:

> A keyed sampler is exactly \(N\)-selection-closed with respect to reference law \(Q_x\) if its key-averaged candidate tuple has law \(Q_x^{\otimes N}\). It is computationally \(N\)-selection-closed if the tuple is computationally indistinguishable from \(Q_x^{\otimes N}\).

Then state selector closure as a theorem or corollary.

## 10.2 Theorem framing

Do not oversell Theorem 1 as a deep mathematical theorem. Present it as identifying the right certification target:

> The sufficiency direction follows from equality of pushforward distributions. The necessity direction shows that any guarantee intended to hold uniformly over all key-independent selectors must determine the full joint candidate law.

## 10.3 Practical selector class

After the theorem, add:

> The universal selector class yields an exact characterization. The positive result directly covers practical inference-time selectors, including hard and soft Best-of-\(N\), reward reranking, verifier selection, rejection sampling, filter-and-sample, and randomized voting rules, provided that the selector does not access the watermark key.

This avoids the impression that the theory relies only on artificial probe selectors.

---

# 11. Reframe the Correlation Analysis as Test-Time Compute Attenuation

The Gaussian result should be tied directly to inference-time scaling.

Current mathematical form:

\[
\mathbb E[\max_i X_i]
=
\mu+\sigma\sqrt{1-\rho}\,m_N.
\]

Interpret it as:

> Candidate correlation reduces the effective test-time scaling gain by a factor of \(\sqrt{1-\rho}\).

Suggested name:

> **Lemma: Correlation attenuates test-time scaling.**

Then explain:

- \(\rho=0\): full independent Best-of-\(N\) gain;
- \(0<\rho<1\): partial loss;
- \(\rho=1\): all scaling gain disappears.

This makes the theory more relevant to alignment and test-time scaling than the phrase “selection bias.”

Potential normalized metric:

\[
\mathrm{ScalingEfficiency}(N)
=
\frac{\mathbb E[R_{\text{selected}}]-\mathbb E[R_{\text{single}}]}
{\mathbb E[R_{\text{independent BoN}}]-\mathbb E[R_{\text{single}}]}.
\]

Under the Gaussian model:

\[
\mathrm{ScalingEfficiency}(N)=\sqrt{1-\rho}.
\]

This is a clean headline quantity.

---

# 12. Rework the Main Experimental Table

The current reward values alone are not maximally intuitive.

## 12.1 Add “BoN gain retained”

Define:

\[
\mathrm{GainRetained}
=
\frac{R_{\mathrm{method},N}-R_{\mathrm{method},1}}
{R_{\mathrm{unwm},N}-R_{\mathrm{unwm},1}}.
\]

This directly reports how much inference-time alignment gain survives watermarking.

Expected qualitative result:

- shared deterministic key: \(0\%\);
- softened shared key: approximately \(100\%\), but without exact marginal preservation;
- candidate-separated subkeys: approximately \(100\%\), with exact marginal and computational joint guarantees.

## 12.2 Recommended main table

| Method | Unique / \(N\) | BoN gain retained | Marginal guarantee | Joint guarantee | Post-selection detection |
|---|---:|---:|---|---|---:|
| Unwatermarked | \(N/N\) | \(100\%\) | reference | exact product | — |
| Shared deterministic key | \(1/N\) | \(0\%\) | exact | no | strong |
| Softened shared key | \(N/N\) | \(\approx100\%\) | altered | no | weaker |
| Independent candidate keys | \(N/N\) | \(\approx100\%\) | exact | exact | oracle/union |
| Master-key subkeys | \(N/N\) | \(\approx100\%\) | exact | computational | strong |

This table communicates the full paper in one glance.

## 12.3 Do not call unwatermarked BoN a ceiling

Use:

- unwatermarked BoN reference;
- matched independent-sampling reference;
- available Best-of-\(N\) gain.

The watermark decoder may have a slightly different variance or sharpness, so “ceiling” is too strong.

## 12.4 Avoid unsupported explanations

Replace:

> the slight excess is the watermark’s variance bonus

with:

> the result is numerically slightly above the unwatermarked reference and within the uncertainty of the current evaluation.

Only use “variance bonus” if it is separately established theoretically or statistically.

---

# 13. Redesign the Main Figure

A TRACE-like paper benefits from one instantly understandable figure.

## Proposed Figure 1: Watermarking meets Best-of-\(N\)

Three horizontal rows:

### Unwatermarked Best-of-\(N\)

\[
x \rightarrow
\{Y_1,Y_2,\ldots,Y_N\}_{\text{i.i.d.}}
\rightarrow
\operatorname*{argmax}_i r(Y_i)
\]

Label:

> Diverse candidates; full test-time alignment gain.

### Shared-key deterministic watermark

\[
x,k \rightarrow
\{Y,Y,\ldots,Y\}
\rightarrow
Y
\]

Label:

> Exact single-sample marginal; no candidate diversity; zero Best-of-\(N\) gain.

### Candidate-separated watermark

\[
x,k_1,\ldots,k_N
\rightarrow
\{Y_1,\ldots,Y_N\}
\rightarrow
\operatorname*{argmax}_i r(Y_i)
\]

Label:

> Product-like candidate law; restored alignment gain; union detection.

On the side, show:

\[
\text{marginal correctness}
\not\Rightarrow
\text{selection correctness}.
\]

This figure is more important than a detailed workflow diagram.

## Proposed Figure 2: Scaling gain versus \(N\)

Plot:

- unwatermarked;
- shared key;
- softened shared key;
- candidate-separated subkeys.

Y-axis:

\[
R_N-R_1
\]

or gain retained.

This makes the test-time scaling framing visually explicit.

---

# 14. Detection Section Revisions

## 14.1 Separate statistical and computational cost

Use:

> The union wrapper introduces two distinct costs. Statistically, testing \(N\) candidate subkeys increases the threshold by \(O(\sqrt{\log N})\). Computationally, detection requires \(N\) evaluations of the base detector, which are independent and parallelizable.

Do not say:

> the only cost is \(\sqrt{2\ln N}\).

## 14.2 More precise finite-\(N\) statement

At significance level \(\alpha\), the threshold correction is more accurately expressed as:

\[
\Phi^{-1}(1-\alpha/N)-\Phi^{-1}(1-\alpha),
\]

under a normal approximation.

Its dominant growth is:

\[
\Theta(\sqrt{\log N}).
\]

Use \(\sqrt{2\log N}\) only as an asymptotic leading order, not a finite-sample exact shift.

## 14.3 Connect detection to alignment selection

Separate:

1. multiplicity loss from searching over subkeys;
2. signal loss caused by reward selection preferring weaker-watermarked candidates.

These are different mechanisms.

Recommended wording:

> Candidate separation does not remove reward–watermark anticorrelation. Even under independent candidate keys, Best-of-\(N\) can prefer candidates with weaker watermark statistics. This selection-induced signal loss is orthogonal to the multiplicity correction of the union detector.

This is already present conceptually, but should be stated more cleanly.

---

# 15. Related Work Reorganization

Organize related work around the interaction with inference-time scaling.

## 15.1 Distortion-free watermarking

Discuss:

- Gumbel/Aaronson;
- inverse-transform methods;
- permute-and-flip;
- robust distortion-free watermarking.

Emphasize that these works primarily analyze a single generation or key-averaged marginal law.

## 15.2 Watermarking and inference-time alignment

Discuss:

- Alignment Resampling;
- WaterMax;
- WaterSearch;
- reward-based selection over watermarked candidates.

Clarify the distinction:

> Some prior works use multi-candidate selection to improve watermark strength or alignment. Our question is whether the original distortion-free guarantee itself is preserved when a separate inference-time selector acts on the candidate pool.

## 15.3 Key reuse and candidate dependence

Discuss prior key-collision work.

Position the paper as:

> Prior work studies how key reuse changes generated-text distributions. We study how candidate dependence interacts with an inference-time selector and changes the post-selection output or eliminates test-time scaling gains.

## 15.4 Multi-key detection

Connect the union wrapper to established multi-key or multi-message detection practices. Do not overclaim detector novelty.

Recommended wording:

> The detector wrapper itself follows a standard multiple-testing principle; our contribution is to show that this wrapper completes the selection-compatible lift and to quantify its cost in the test-time selection setting.

This makes the paper more honest and stronger.

---

# 16. Experimental Revision Priorities

## Priority 1: Independent-key oracle baseline

Add a separate arm with independently sampled candidate keys.

Purpose:

- demonstrates exact joint closure;
- cleanly separates exact from computational construction;
- establishes the PRF construction as a compact realization rather than the only possible solution;
- removes the false impossibility framing.

Compare:

1. shared key;
2. independent stored keys;
3. PRF-derived subkeys;
4. softened shared key;
5. unwatermarked reference.

## Priority 2: More inference-time selectors

The current selector family is already useful. Center it more explicitly around alignment/test-time scaling:

- hard Best-of-\(N\);
- soft Best-of-\(N\);
- ExpBoN;
- verifier thresholding;
- rejection sampling;
- filter-and-sample;
- self-consistency or majority voting;
- process-reward-model selection, if available.

The point is not merely selector breadth. It is to validate:

> The same joint-law issue appears across common inference-time alignment mechanisms.

## Priority 3: More values of \(N\)

Use:

\[
N\in\{1,2,4,8,16,32\}
\]

if computationally feasible.

Report:

- reward gain;
- gain retained;
- unique fraction;
- pairwise similarity;
- detection;
- union threshold;
- detector latency.

This strengthens the “test-time scaling” narrative.

## Priority 4: More generator/reward combinations

At minimum:

- one additional instruction-tuned model family;
- one math/reasoning-oriented model if possible;
- one additional reward model or verifier.

This answers whether the failure is specific to HH-RLHF and ArmoRM.

A valuable reasoning setup would be:

- math prompts;
- candidates selected by answer verifier or process reward;
- measure pass@1 / selected accuracy rather than only scalar preference reward.

This would connect the paper strongly to inference-time scaling.

## Priority 5: Self-consistency experiment

Generate multiple watermarked reasoning traces and aggregate final answers by majority vote.

Compare:

- independent unwatermarked traces;
- shared-key watermarked traces;
- candidate-separated watermarked traces.

Expected story:

- shared-key deterministic watermark reduces the effective number of reasoning paths;
- candidate-separated keys restore self-consistency gains.

This is highly aligned with the paper’s general theorem and broadens the story beyond reward-model BoN.

## Priority 6: Statistical reporting

For all headline reward comparisons, provide:

- confidence intervals;
- paired bootstrap over prompts;
- significance or equivalence intervals where appropriate.

Avoid claiming strict reward dominance based on small numerical differences without uncertainty.

---

# 17. Suggested Section Structure

## 1. Introduction

- Test-time alignment generates and selects candidates.
- Distortion-free watermarking is marginal.
- Marginal guarantees do not compose with selection.
- Shared-key collapse.
- General joint-law issue.
- Contributions.

## 2. Background and Related Work

- Distortion-free watermarking.
- Best-of-\(N\) and test-time scaling.
- Watermarking under alignment selection.
- Key reuse and multi-key detection.

## 3. When Watermarking Eliminates Test-Time Scaling

- Formal setup.
- Shared-key collapse theorem.
- Coin example.
- Partial-correlation model.
- Scaling-gain attenuation.

## 4. Selection-Closed Distortion-Freeness

- Exact definition.
- Computational definition.
- Characterization for key-independent selectors.
- Guarantee hierarchy.

## 5. Candidate-Separated Watermarking

- Exact independent-key lift.
- Single-master-key PRF realization.
- Applicability conditions.
- Union detector.
- Statistical and computational costs.

## 6. Experiments

- Best-of-\(N\) alignment.
- Scaling with \(N\).
- Selector family.
- Detection after selection.
- Second watermark family.
- Self-consistency/verifier setting.
- Runtime and calibration.

## 7. Limitations

- Key-dependent selectors are outside the guarantee.
- Candidate separation does not remove reward–watermark anticorrelation.
- Computational closure is weaker than adaptive multi-query undetectability.
- Union detection scales linearly in the number of candidate subkeys.
- The study focuses on deterministic keyed decoder families.

## 8. Conclusion

End on:

> Watermark guarantees intended for test-time alignment must be stated at the level of candidate sets, not isolated generations.

---

# 18. Recommended Guarantee Hierarchy

Use a small figure or proposition:

\[
\text{adaptive multi-query undetectability}
\Longrightarrow
\text{computational selection closure}
\Longrightarrow
\text{marginal distortion-freeness}.
\]

Also distinguish exact independent-key closure from computational master-key closure.

Example table:

| Property | Controls one sample | Controls Best-of-\(N\) | Controls repeated adaptive queries |
|---|---:|---:|---:|
| Marginal distortion-freeness | yes | no | no |
| Exact \(N\)-selection closure | yes | yes for fixed \(N\) | not necessarily |
| Computational \(N\)-selection closure | yes | yes for efficient selectors | not necessarily |
| Adaptive multi-query undetectability | yes | yes | yes |

Do not claim that selection closure is equivalent to full watermark undetectability.

---

# 19. Specific Language Replacements

## Replace “deployed output”

Use:

> post-selection output

## Replace “deployment-level guarantee”

Use:

> selection-level guarantee

or:

> inference-time guarantee

## Replace “deployed systems increasingly ship selection”

Use:

> modern inference-time alignment increasingly relies on candidate generation and selection.

## Replace “the deployed shared-key scheme”

Use:

> the shared-key deterministic instantiation.

## Replace “the exact scheme family that is deployed today”

Use:

> the deterministic context-hashed scheme family commonly studied and used in current watermarking systems.

Only retain “used” if supported by citations.

## Replace “one-line fix”

Use:

> minimal lift

## Replace “strictly distortion-free”

Use:

> marginally distortion-free with computational selection closure.

## Replace “unwatermarked ceiling”

Use:

> matched unwatermarked Best-of-\(N\) reference.

## Replace “strictly dominates”

Use:

> improves the reward–detection–distribution trade-off

unless statistical strict dominance is demonstrated.

---

# 20. Rewritten Discussion Points

## 20.1 Main implication

> Test-time compute scaling assumes access to multiple useful candidate draws. A watermark can preserve every candidate marginal while changing the dependence structure of the pool, thereby changing the return from additional inference-time compute.

This is a strong general statement.

## 20.2 Implication for alignment

> Watermarking and alignment should not be evaluated independently when alignment is performed through selection. A watermarking decoder may appear quality-preserving at \(N=1\) but substantially alter the reward or accuracy scaling curve as \(N\) increases.

## 20.3 Implication for watermark certification

> A single-sample distortion-free certificate is insufficient when the watermarked model will be combined with Best-of-\(N\), reranking, verification, or self-consistency. The certificate should state whether candidate-level randomness is independent and whether the candidate joint law matches the intended reference.

## 20.4 Scope

> The paper does not claim to solve all interactions between watermarking and alignment. Candidate separation restores the intended candidate law, but reward selection may still reduce watermark detectability by preferring candidates with weaker watermark evidence.

---

# 21. Proposed Conclusion

Distortion-free watermarking is usually defined for one generation, while inference-time alignment and test-time scaling act on a set of generations. We show that this distinction is consequential: exact candidate marginals can coexist with a correlated joint law that changes the post-selection distribution, and shared deterministic randomness can eliminate the entire Best-of-\(N\) gain. The appropriate guarantee is therefore selection-level and joint: the candidate pool should match independent samples from the reference model.

Candidate-specific keys provide a minimal correction. Independent keys give exact selection closure, while PRF-derived subkeys give a compact computational realization under one master key. Together with a union detector, this restores the reward and diversity benefits of selection-based inference while retaining the base watermark’s marginal guarantee.

The broader lesson is that watermarking should be evaluated not only at \(N=1\), but along the same test-time scaling curves used for alignment. A watermark is compatible with Best-of-\(N\) only when its guarantee survives selection.

---

# 22. High-Priority Revision Checklist

## Must fix before submission

- [ ] Remove all unsupported claims that exact distortion-freeness and detectability are incompatible.
- [ ] Separate exact independent-key closure from computational PRF-derived closure.
- [ ] Replace “strictly distortion-free” where it incorrectly describes the joint PRF construction.
- [ ] Replace “the only cost” with separate statistical and computational costs.
- [ ] Remove or qualify “deployed” language.
- [ ] Reframe the paper around watermarking under alignment and test-time scaling.
- [ ] Add the “not merely a seeding issue” paragraph.
- [ ] Reorganize contributions so that the joint guarantee is primary and subkeys are the minimal realization.
- [ ] Replace “unwatermarked ceiling” with “unwatermarked reference.”
- [ ] Remove “variance bonus” unless supported.

## Strongly recommended

- [ ] Add independent candidate keys as an exact oracle baseline.
- [ ] Add gain-retained or scaling-efficiency as a headline metric.
- [ ] Add confidence intervals to reward comparisons.
- [ ] Add a reasoning/verifier or self-consistency experiment.
- [ ] Add at least one additional generator family.
- [ ] Add a simple main figure contrasting independent, shared-key, and candidate-separated pools.
- [ ] Report detector latency and \(N\)-scaling separately from the statistical threshold penalty.

## Nice to have

- [ ] Rename JCN to a term that more directly indicates selection closure.
- [ ] Add a guarantee hierarchy figure.
- [ ] Add a short discussion of adaptive or key-dependent selectors.
- [ ] Add an explicit applicability statement for keyed decoders whose randomness can be domain-separated.

---

# 23. Final Positioning

The paper should be remembered as:

> **The first systematic study of whether distortion-free watermarking is preserved under selection-based inference.**

Not as:

> A paper that notices identical candidates and changes the random seed.

The core scientific sequence should be:

\[
\boxed{
\text{marginal watermark guarantee}
\;\not\Rightarrow\;
\text{correct test-time selection}
}
\]

\[
\boxed{
\text{joint candidate preservation}
\;\Longleftrightarrow\;
\text{selector-universal closure}
}
\]

\[
\boxed{
\text{candidate-separated keys}
\;\Rightarrow\;
\text{restored Best-of-}N\text{ and alignment scaling}
}
\]

That is the strongest and most defensible ICLR framing.
