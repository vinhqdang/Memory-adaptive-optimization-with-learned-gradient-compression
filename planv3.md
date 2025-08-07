### Repository status summary (checked 07 Aug 2025, 14:10 UTC)

| Plan v2 work‑package / task                           | Evidence in latest push                                                                                                                                                                 | ✅ / ⚠️                  |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **WP‑1 Algorithmic enhancements**                     | `mango/mango_lrq.py`, `tinyformer_policy.py`, NF4 & bits‑and‑bytes added to `environment.yml` and commit log “Implement MANGO‑LRQ … TinyFormer … NF4 quantization” ([GitHub][1])        | **✅ implemented**       |
|  T1.1  8‑bit / NF4 states                             | NF4 quantisation implemented; 8‑bit momentum *mentioned* in code header but not obvious in diff ⇒ needs confirm in `enhanced_optimizer.py`                                              | ⚠️ **partially**        |
|  T1.2  EF21 residual buffer                           | Commit log says “variance‑reduced error compensation (Byz‑VR‑MARINA inspired)” ([GitHub][1]) – EF mechanism seems merged, but naming differs ⇒ review function `apply_error_feedback()` | ⚠️ likely               |
|  T1.3  Multi‑objective reward incl. wall‑clock/energy | Memory & time tracking (`memory_profiler.py`) in tree, but **no** power‑draw sampler yet; reward vector in `tinyformer_policy.py` appears 2‑dim (loss, mem)                             | ⚠️ **incomplete**       |
| **WP‑2 Theory** (α‑contractive proof, Pareto bounds)  | No `docs/`, `proof.md` or appendix files added; commit message makes no reference                                                                                                       | ❌ **missing**           |
| **WP‑3 Experiments**                                  | New scripts: `experiments/baseline_optimizers.py`, `experiments/imagenet_experiment.py`, `demo_mango_lrq.py` ([GitHub][1])                                                              | **✅ basic scaffolding** |
|  • CIFAR‑10 / ImageNet runs                           | Ready‑to‑run scripts exist but logs/plots not committed → still need results upload                                                                                                     | ⚠️ pending results      |
|  • LLaMA‑7B LoRA                                      | **No** `llama_experiment.py` or LoRA hooks found                                                                                                                                        | ❌ not started           |
| **WP‑4 System integration**                           | `memory_profiler.py` present; **no** ZeRO‑3/FSDP integration files (`mango_sharder.py`) yet                                                                                             | ❌                       |
| **WP‑5 Writing / reproducibility**                    | `planv2.md` is in repo root ([GitHub][2]); no draft manuscript, no docker/Colab                                                                                                         | ⚠️ early stage          |

---

### Priority action list

1. **Confirm 8‑bit optimizer states**
   *Search `enhanced_optimizer.py` for calls to `torch.int8` or `bnb.nn.optim.Adam8bit`.* If only NF4 on gradients, add momentum quantisation via the bits‑and‑bytes “optim8bit” wrapper.

2. **Finish multi‑objective reward (energy)**
   *Add an async thread that samples `nvidia‑smi --query-gpu=power.draw --format=csv,noheader,nounits` every Δt and writes to a shared queue; the TinyFormer policy can then read an exponentially‑weighted mean.*

3. **ZeRO‑3 / FSDP plumbing**
   *Create `mango/zero_utils.py` that auto‑generates ZeRO parameter groups from the policy’s compression decisions, and register a custom reducer hook for FSDP.*

4. **Theoretical appendix**
   *Start a `docs/` folder with `theory.md`; port the α‑contractive lemma and VR‑MARINA variance bound exactly as outlined in §3.3 of plan v2.*

5. **End‑to‑end runs & artefacts**
   *Run the ImageNet script with `--log_wandb`, export peak/median memory CSV, and push W\&B run link or raw logs under `results/`.*

6. **LoRA experiment skeleton**
   *Duplicate `imagenet_experiment.py` ➜ `llama_lora_experiment.py`; use `transformers` + `bitsandbytes` 4‑bit weights and keep LoRA adapters uncompressed.*

7. **Reproducibility/delivery**
   *Add `Dockerfile` (Ubuntu 22.04 + CUDA 12.2 + PyTorch 2.5), update README with one‑liner:
   `docker run --gpus all -v $PWD:/workspace mango:v2 python demo_mango_lrq.py`.*

---

### Quick wins before the next push

| Effort    | Win                                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **\~1 h** | Rename Byz‑VR code path to `ef21_buffer.py` and add citation in doc‑string (reviewer friendliness).                            |
| **\~2 h** | Add a `--energy` flag to reward config; log power+time to W\&B – counts toward O4 with minimal code.                           |
| **\~3 h** | Stub `theory.md` with theorem statements and placeholders; even partial proofs look better than none in an initial submission. |

Once the above are in place, the repository will cover *all* WP‑1 deliverables and most of WP‑3, leaving only theory polish and distributed integration as substantial chunks before the October draft freeze.

[1]: https://github.com/vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression/commit/c77e97fe7e788099654439ca017c1a2b5d74461f "Implement MANGO-LRQ: Enhanced Memory-Adaptive Neural Gradient Optimizer · vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression@c77e97f · GitHub"
[2]: https://github.com/vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression "GitHub - vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression"
