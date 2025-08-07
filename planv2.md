Below is a consolidated technical review of **MANGO — Memory‑Adaptive Neural Gradient Optimizer** together with concrete, ICLR‑style recommendations drawn from the most relevant optimization and gradient‑compression papers published at ICLR (and closely‑related venues) since 2020.

---

## 1  Current strengths of MANGO

| Aspect                                | Evidence in repo                                                                                                                                                        | Why it already looks promising                                              |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Truly adaptive budget**             | Optimizer exposes a `memory_budget` parameter and an LSTM policy that can choose bit‑width, sparsity level and whether to store momentum in low precision ([GitHub][1]) | Few published methods *learn* when to compress; most fix a scheme a‑priori. |
| **Policy‑gradient training (PPO)**    | `mango/ppo_trainer.py`, described in README ([GitHub][1])                                                                                                               | Reinforcement‑learning control of compression is novel in this space.       |
| **Error‑feedback & statistics hooks** | Compressor returns residuals; stats collector tracks running norms ([GitHub][1])                                                                                        | Helps recover accuracy lost by aggressive compression.                      |

These are good foundations, but the implementation and evaluation **fall short of today’s ICLR bar** in four critical areas (scale, theory, baselines, and clarity). The sections that follow give targeted upgrades.

---

## 2  Key trends in ICLR 2020‑2025 you must address

| Trend                                                       | Representative paper(s)                                                                                                              | Relevance to MANGO                                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Low‑rank *gradient* projection for memory**               | **GaLore** (2024) shows up to 63 % total‑memory saving on 7 B‑param LLaMA pre‑training ([arXiv][2])                                  | Demonstrates that rank‑reduction alone can beat quantization‑only schemes; MANGO currently ignores low‑rank structure. |
| **Adaptive rank scheduling**                                | **AdaRankGrad** (ICLR 2025) gradually *reduces* gradient rank during training ([OpenReview][3])                                      | Reinforces the value of *dynamic* policies; provides comparison point for your RL controller.                          |
| **Fine‑grain gradient quantization informed by statistics** | “Neural gradients are near‑lognormal” (ICLR 2021) achieves loss‑less 6‑bit gradients using closed‑form scaling ([ICLR][4])           | Suggests statistical features your policy network should ingest.                                                       |
| **Compression + convergence proofs**                        | Byz‑VR‑MARINA (ICLR 2023) couples variance‑reduction with compressed messages and supplies tight theoretical rates ([OpenReview][5]) | ICLR reviewers increasingly expect *some* guarantees when compression is adaptive.                                     |
| **Quantization/optimizer paging for LLMs**                  | **QLoRA** (NeurIPS 2023) introduced NF4 + paged optimizers to cut memory by 4× ([arXiv][6])                                          | Shows that bits‑per‑state can be <8 even for 65 B models—your method should at least match that.                       |

---

## 3  Algorithmic upgrades to include

> **Goal** – position the submission as *strictly more general* than GaLore, AdaRankGrad and QLoRA while retaining MANGO’s reinforcement‑learning flavour.

### 3.1 Hybrid Low‑Rank + Quantized Compression (MANGO‑LRQ)

1. **Two‑stage approximation at every step**

   1. Compute rank‑r projection *à la* GaLore (randomized SVD, r∈{1,4,8}).
   2. Quantize the factor matrices $P,Q$ to 4‑ or 8‑bit NF4 using per‑column scales (matching QLoRA).
2. **RL policy outputs** a discrete triple $(r, b_P, b_Q)$ plus a switch for momentum precision.
3. **Memory estimator** feeds policy with *peak* CUDA‑memory since last update ⇒ encourages stable training rather than instantaneous savings.
4. **Error‑feedback** accumulates residuals *in full precision* but down‑casts them lazily with the same policy parameters (keeps theory tidy).

### 3.2 Variance‑Reduced, Error‑Compensated Updates

*Borrow* the Byz‑VR‑MARINA decomposition: keep an uncompressed reference gradient every $T$ steps, apply variance‑reduction to the compressed updates in between, and carry MANGO’s residual vector separately.
This directly yields an $O(\sigma^2/r)$ or $O(\sigma^2 b^{-2})$ term in the convergence proof and sidesteps the reviewers’ usual “what happens when compression noise explodes?” question.

### 3.3 Lightweight theory sketch

Provide a 1‑page appendix proving **$\mathcal O(1/\sqrt{T})$** convergence for non‑convex loss with *adaptive but bounded* compression error:

$$
\mathbb E \|\nabla f(\theta_\text{avg})\|^2
\le
\frac{2(f(\theta_0)-f^\star)}{\eta T}
+\underbrace{\frac{L \eta \sigma^2}{m}\bigl(1+\delta_\text{comp}\bigr)}_{\text{compression penalty}}
$$

where $\delta_\text{comp}\le c\,(r^{-1}+2^{-b})$ is controlled by the policy network.  Cite GaLore for the low‑rank bound and Byz‑VR‑MARINA for the variance‑reduced part.

---

## 4  Experimental plan that will satisfy ICLR reviewers

| Dimension                                                          | Minimum required upgrade                                                                                                                | Rationale                                                   |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Scale**                                                          | Full‑parameter pre‑training of 1 B and 7 B LLaMA on 8× A100 (80 GB) and *single* RTX 4090 (24 GB) with policy‑controlled memory budget. | Direct comparison to GaLore’s headline result ([arXiv][2]). |
| **Vision**                                                         | ImageNet‑1k with ResNet‑50 and ViT‑B; include 4‑bit gradient + rank‑4 low‑rank vs plain Adam/Adafactor.                                 | Matches prior “Neural gradients near‑lognormal” study.      |
| **Baselines**                                                      | AdamW, Adafactor, GaLore (low‑rank only), AdaRankGrad (rank schedule), QLoRA pager (quantization only).                                 | Shows you improve on *all* three paradigms.                 |
| **Metrics**                                                        | • Final top‑1 / perplexity                                                                                                              |                                                             |
| • *Peak* and *median* GPU memory (track with PyTorch 2.5 profiler) |                                                                                                                                         |                                                             |
| • Throughput (tokens/sec, imgs/sec)                                |                                                                                                                                         |                                                             |
| • Time‑to‑27‑bleu (NMT) or 80 % ImageNet.                          | ICLR reviewers care about *speed* and *memory*, not just accuracy.                                                                      |                                                             |
| **Ablations**                                                      | (i) RL vs heuristic schedule, (ii) with/without low‑rank stage, (iii) error‑feedback variants.                                          | Separates contributions.                                    |
| **Policy analysis**                                                | Plot rank/bit‑width decisions over training; correlate with loss curvature (trace of Hessian via Hutch++).                              | Provides insight → easier acceptance.                       |

---

## 5  Manuscript structure & clarity checklist

1. **Motivation in first page**: “We unify low‑rank projection **and** learned quantization under a single RL controller.”
2. **Three bullet contributions** (in bold) immediately after motivation.
3. **Algorithm 1** box with <15 lines pseudocode; keep math in appendix.
4. **Figure 2**: Memory vs accuracy Pareto curve (MANGO‑LRQ dominates).
5. **Social‑impact & limitations**: discuss energy savings and potential degradation on fairness tasks.
6. **Reproducibility**: commit hash, exact command lines, hardware, random seeds; supply docker.
7. **Theory appendix** directly cites Theorem 1 of GaLore and adapts Lemma 3 of Byz‑VR‑MARINA.

---

## 6  Additional low‑effort, high‑value tweaks

* **Replace policy LSTM with TinyFormer (6‑layer, 128‑d) –** better long‑range trend capture, still tiny (<100 k params).
* **Use NF4 quantization code from bits‑and‑bytes** for the 4‑bit stage (drop‑in, battle‑tested).
* **Automatic mixed‑precision (AMP) everywhere except final gradient cast** – no reviewer complaints about apples‑to‑oranges speed comparisons.
* **Provide a Colab notebook** that trains ResNet‑18 on CIFAR‑10 under 2 GB VRAM to showcase memory adaptivity.

---

## 7  Summary: positioning for ICLR 2026

With the **MANGO‑LRQ** extension, strong baselines at *modern* scale, a short but solid convergence analysis, and clear empirical wins on the memory‑accuracy trade‑off, the work will:

* subsume GaLore’s low‑rank idea,
* outperform AdaRankGrad by choosing both rank *and* precision, and
* exceed QLoRA by acting on gradients *and* optimizer states.

That combination, together with MANGO’s RL‑based novelty, should make the paper competitive for acceptance in the Optimization or Efficient Training tracks at ICLR 2026.

Good luck—feel free to ask for a focused review of your revised draft once the new experiments or theory sections are in place!

[1]: https://github.com/vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression "GitHub - vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression"
[2]: https://arxiv.org/abs/2403.03507 "[2403.03507] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
[3]: https://openreview.net/forum?id=LvNROciCne "AdaRankGrad: Adaptive Gradient Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning | OpenReview"
[4]: https://iclr.cc/virtual/2021/poster/2716 "ICLR Poster Neural gradients are near-lognormal: improved quantized  and sparse training"
[5]: https://openreview.net/forum?id=pfuqQQCB34 "Variance Reduction is an Antidote to Byzantines: Better Rates, Weaker Assumptions and Communication Compression as a Cherry on the Top | OpenReview"
[6]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"

