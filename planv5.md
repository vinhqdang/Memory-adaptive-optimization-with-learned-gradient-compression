### Repository audit – pushed commit `56bbe41` (07 Aug 2025 16:10 UTC)

I re‑checked the public *main* branch on GitHub and found **eight commits** (commit‑hash `56bbe413…`).
Two successive commits are especially relevant:

| Commit                                                                                      | What it adds                                                                                                                                                                                                                                                           | Gaps closed (plan v4 IDs)                                                                             |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **`ac3e06b` – “Implement MANGO‑LRQ v4.0: 8‑bit States + Power‑Aware Optim.”** ([GitHub][1]) | \* `mango/enhanced_optimizer.py` – integrates `bitsandbytes.optim.Adam8bit`<br>\* `utils/power_sampler.py` + policy hooks → energy term<br>\* `mango/ef21_buffer.py` (renamed, cited)<br>\* unit‑tests `tests/test_8bit_state.py`, `test_power_sampler_integration.py` | **G1 8‑bit state** ✔︎<br>**G2 EF21 docs** ✔︎<br>**G3 energy reward** ✔︎<br>**G8 smoke/unit tests** ✔︎ |
| **`56bbe41` – “Complete planv4.md Implementation: Final Gap‑Closing Tasks”** ([GitHub][2])  | \* `planv4.md` committed<br>\* CI pipeline `.github/workflows/ci.yml`<br>\* CSV artifacts under `results/` (`imagenet_peakmem.csv`, `cifar10_peakmem.csv`)                                                                                                             | **G5 experiment logs** ✔︎<br>**G10 CI smoke test** ✔︎                                                 |

---

#### Current status versus *plan v4*

| Gap ID                                    | Status after the latest push                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **G1 8‑bit momentum/state**               | **Closed** – verified in `enhanced_optimizer.py` ([GitHub][1])                                                            |
| **G2 EF21 naming + citation**             | **Closed** – file renamed & cited (commit diff) ([GitHub][1])                                                             |
| **G3 Energy‑aware reward**                | **Closed** – `utils/power_sampler.py` & TinyFormer update ([GitHub][1])                                                   |
| **G4 Theory appendix** (`docs/theory.md`) | **Still missing** – `docs/` folder exists but no `theory.md` was created in either commit; GitHub search returns nothing. |
| **G5 Empirical CSVs**                     | **Closed** – two peak‑memory CSVs present ([GitHub][2])                                                                   |
| **G6 LLaMA‑LoRA experiment script**       | **Not yet found** – `experiments/` contains ImageNet & baseline scripts only.                                             |
| **G7 ZeRO‑3 / FSDP integration**          | **Still missing** – no `mango/zero_utils.py` or `fsdp_hook.py` in tree.                                                   |
| **G8 Imitation warm‑start script**        | **Still missing** – no `scripts/collect_comp_ams.py`.                                                                     |
| **G9 Central YAML config**                | **Still missing** – no `configs/` directory yet.                                                                          |
| **G10 CI workflow**                       | **Closed** – GitHub Actions smoke test in place ([GitHub][2])                                                             |
| **plan v4 progress table**                | **Out‑of‑date** – plan still lists G1–G3, G5 & G10 as “missing”. Needs check‑marks & new due‑dates.                       |

---

### Immediate follow‑ups

1. **Add `docs/theory.md`** with Theorem 1/Lemma 2 (closes **G4**).
2. **Stage `experiments/llama_lora_experiment.py`** (closes **G6**).
3. **Commit distributed helpers** `mango/zero_utils.py`, `mango/fsdp_hook.py` (**G7**).
4. **Push imitation‑data script** (`scripts/collect_comp_ams.py`) (**G8**).
5. **Introduce `configs/default.yaml`** and load logic in `main.py` (**G9**).
6. **Update the status table in `planv4.md`** so closed items are ticked and new deadlines set.


[1]: https://github.com/vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression/commit/ac3e06b "Implement MANGO-LRQ v4.0: Complete 8-bit States & Power-Aware Optimiz… · vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression@ac3e06b · GitHub"
[2]: https://github.com/vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression/commit/56bbe413ab0a85f7780cda92caa62d86f739c3af "Complete planv4.md Implementation: Final Gap-Closing Tasks · vinhqdang/Memory-adaptive-optimization-with-learned-gradient-compression@56bbe41 · GitHub"
