# Kronos Codebase Review Report

Review target: `D:\Programming\AiProjects\Kronos(reviewing)\Kronos`  
Review date: 2026-06-05  
Reviewer context: comparing Kronos against MMFPS_GEN_V2 financial diffusion research.

## Executive Summary

Kronos is genuinely valuable for your research, but not because it should replace MMFPS. Its core idea is different:

- MMFPS models continuous future return paths with conditional diffusion.
- Kronos models financial candles as a discrete financial language using a tokenizer plus autoregressive Transformer.

The goldmine is the representation strategy:

1. Continuous OHLCV data is converted into hierarchical discrete tokens.
2. A decoder-only Transformer learns next-token dynamics.
3. Future candles are generated autoregressively through stochastic sampling.
4. Temporal embeddings are built directly into the model.
5. Coarse/fine token prediction creates a two-level market representation.

For MMFPS, the best takeaway is not "switch to Kronos." The best takeaway is:

> Add hierarchical market structure and coarse-to-fine path thinking later, after diffusion emergence is stable.

Kronos is architecture-rich, research-useful, and conceptually close to market manifold modeling. But its public inference interface averages stochastic samples, several training/evaluation paths are demo-grade, and at least one CSV fine-tuning path leaks future information through normalization.

## Project Shape

Project-owned Python files, excluding `.venv`: 31

Important folders:

- `model/`: core tokenizer, Transformer predictor, inference wrapper.
- `finetune/`: Qlib-based tokenizer/predictor fine-tuning and backtesting.
- `finetune_csv/`: CSV fine-tuning path.
- `examples/`: prediction and backtesting demos.
- `tests/`: regression tests against fixed Hugging Face model revisions.
- `webui/`: Flask/Plotly local demo around live BTC prediction.

## Core Architecture

### 1. KronosTokenizer

File: `model/kronos.py`

`KronosTokenizer` is an encoder/decoder tokenizer for continuous K-line features.

Flow:

```text
OHLCV continuous features
  -> linear embedding
  -> Transformer encoder
  -> quantization projection
  -> Binary Spherical Quantizer
  -> hierarchical token IDs
  -> Transformer decoder reconstruction
```

The tokenizer outputs two token levels:

- `s1`: coarse/pre token.
- `s2`: finer/post token.

This is the most important architectural idea for MMFPS. It gives the model a discrete hierarchy for market behavior: broad shape first, detailed realization second.

### 2. Binary Spherical Quantization

File: `model/module.py`

The quantizer converts continuous latent vectors into binary codebook-like representations. It includes entropy and commitment terms, which are intended to keep the code usage alive rather than collapsing into a few codes.

This is relevant to MMFPS as an auxiliary diagnostic idea:

- learn discrete behavior codes from real paths,
- classify generated paths into behavior bins,
- measure whether 128 futures cover multiple behavior codes.

It should not be added directly to MMFPS training yet.

### 3. Kronos Predictor Model

File: `model/kronos.py`

`Kronos` is a decoder-only Transformer:

```text
hierarchical token embedding
  + temporal embedding
  -> causal Transformer blocks
  -> dependency-aware prediction head
  -> s1 logits and s2 logits
```

Notable pieces:

- `HierarchicalEmbedding`: combines `s1` and `s2` token embeddings.
- `TemporalEmbedding`: minute/hour/weekday/day/month information.
- RoPE attention.
- `DependencyAwareLayer`: predicts fine token `s2` conditioned on the coarse token `s1`.
- `DualHead`: separate CE heads for coarse/fine tokens.

This is a good model-language analogy for markets. It treats a candle stream as a sequence of symbols instead of a raw regression target.

### 4. Autoregressive Inference

File: `model/kronos.py`

Kronos rolls forward one future step at a time:

```text
context tokens
  -> sample next coarse token s1
  -> sample next fine token s2 conditioned on s1
  -> append tokens to rolling context buffer
  -> decode generated tokens back to OHLCV
```

This is conceptually similar to MMFPS sampling multiple possible futures, but it is not diffusion. It is next-token market-language generation.

## The Big Similarity To MMFPS

Both systems are trying to model:

```text
P(future market behavior | historical context)
```

Both care about:

- stochastic futures,
- regime branching,
- path structure,
- uncertainty,
- realistic market dynamics,
- conditioning on historical context.

Kronos is therefore aligned with your research direction.

## The Big Difference From MMFPS

MMFPS:

```text
continuous context features -> diffusion denoiser -> 128 future return paths
```

Kronos:

```text
continuous OHLCV -> hierarchical tokens -> autoregressive token LM -> decoded future candles
```

MMFPS tries to learn a continuous conditional future manifold. Kronos turns market data into a language modeling problem.

That difference matters. Kronos’s representation ideas are valuable, but copying its objective would move MMFPS away from diffusion.

## Critical Findings

### Critical 1: Public inference averages stochastic samples

File: `model/kronos.py`

The inference path repeats each input by `sample_count`, generates multiple samples, reshapes to `(batch, sample_count, seq, features)`, then averages:

```python
preds = np.mean(preds, axis=1)
```

This is dangerous for your MMFPS goal.

For your project, the 128 paths are the product. Averaging them destroys:

- path identity,
- branching,
- tail behavior,
- multimodality,
- closest-path containment analysis.

Kronos exposes probabilistic sampling but the default predictor interface collapses it into an ensemble mean. If adapting anything from Kronos, keep samples unaveraged.

### Critical 2: CSV fine-tuning normalization leaks future information

File: `finetune_csv/finetune_base_model.py`

`CustomKlineDataset.__getitem__` builds a full window of:

```text
lookback + predict_window + 1
```

Then computes:

```python
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
```

That uses the entire window, including future target region, to normalize the sample.

This is a leakage bug for forecasting-style evaluation and fine-tuning. The Qlib dataset path does this correctly using only past context, but the CSV path does not.

### Critical 3: Qlib dataset ignores the DataLoader index

File: `finetune/dataset.py`

`QlibDataset.__getitem__(idx)` ignores `idx` and samples randomly:

```python
random_idx = self.py_rng.randint(0, len(self.indices) - 1)
```

This means:

- `DistributedSampler` partitioning is weakened.
- Workers may duplicate samples depending on RNG behavior.
- Epoch coverage is not cleanly auditable.
- Validation is stochastic instead of deterministic.

This is not necessarily fatal for large-scale pretraining, but it is not ideal for controlled evaluation.

### Critical 4: Train/val/test overlap in default Qlib config is broad

File: `finetune/config.py`

Configured splits:

```text
train: 2011-01-01 to 2022-12-31
val:   2022-09-01 to 2024-06-30
test:  2024-04-01 to 2025-06-05
```

The README suggests overlap exists for lookback buffer, but these overlaps are much larger than a normal 90-day lookback buffer. This can be defensible only if preprocessing/evaluation handles boundaries very carefully. As written, it deserves audit before trusting backtest metrics.

### Critical 5: Cross-attention causal behavior changes between train and eval

File: `model/module.py`

`MultiHeadCrossAttentionWithRoPE` sets:

```python
is_causal_flag = self.training
```

So cross-attention is causal during training and non-causal during eval. This may be intentional for sibling-token conditioning, but it is surprising and should be verified. Train/eval attention semantics should not quietly differ unless the design explicitly requires it.

### Critical 6: Metrics are not generative-finance metrics

The core training metrics are reconstruction loss or token CE loss. Regression tests check deterministic output stability and MSE against fixed fixtures.

They do not deeply measure:

- stochastic diversity,
- coverage of actual future by sample set,
- volatility clustering,
- tail realism,
- path structure,
- autocorrelation decay,
- drawdown distribution,
- regime persistence.

This is exactly where MMFPS’s emergence dashboard work is stronger.

### Critical 7: Web UI and live BTC demo are useful but not research-grade

Files:

- `webui/app.py`
- `btc_live_test.py`

The Web UI is good for demonstration, but it uses live Binance BTC data and default predictor outputs. It is not a controlled emergence/evaluation system. It also inherits the predictor’s sample-averaging behavior.

### Critical 8: Comet defaults and path placeholders are brittle

File: `finetune/config.py`

The default config enables Comet:

```python
self.use_comet = True
```

The API/workspace/project values are placeholders. This is not a model bug, but it makes first-run reproducibility more fragile.

## Strengths

Kronos has a much more mature representation stack than a simple time-series regressor.

Strong ideas:

- Market-specific tokenizer.
- Hierarchical coarse/fine tokenization.
- Autoregressive sequence modeling over market tokens.
- Time/calendar embeddings.
- RoPE causal attention.
- Separate tokenizer and predictor training phases.
- Hugging Face model loading.
- Regression tests pinned to model revisions.
- Qlib fine-tuning and backtest example.

Architecturally, it is credible and sophisticated.

## Weaknesses

The main weaknesses are not the model idea. They are in evaluation and some data paths:

- Stochastic samples are averaged by the high-level predictor.
- CSV fine-tuning path has future normalization leakage.
- Qlib sampling is stochastic even for validation.
- Split overlap deserves stricter leakage audit.
- Backtest examples are simplified demos.
- Financial realism metrics are thin.
- The UI is demo-oriented, not emergence-oriented.

## Readiness Assessment

Research architecture: high  
Representation quality: high  
Direct production trading readiness: low  
Evaluation rigor: medium-low  
Usefulness to MMFPS: high as inspiration, low as a drop-in replacement  

Kronos is a serious research codebase, but the public repo should be treated as a foundation-model demo and fine-tuning scaffold, not a fully leakage-proof quantitative research environment.

## What MMFPS Should Borrow

### Borrow Later: Hierarchical future semantics

Kronos’s coarse/fine token split maps beautifully to future path generation:

```text
coarse future behavior:
  direction, volatility envelope, broad trend/range structure

fine future behavior:
  local oscillation, shocks, microstructure, bar-to-bar texture
```

For MMFPS, this could eventually become:

```text
coarse diffusion path / envelope
  -> residual diffusion path
```

But do not add this until pure diffusion and Phase B structural losses are understood.

### Borrow Soon: Temporal embeddings

Kronos explicitly embeds:

- minute,
- hour,
- weekday,
- day,
- month.

MMFPS already has session-aware data. The low-risk adaptation is better explicit time/session embeddings in conditioning, not a model rewrite.

### Borrow Soon: Unaveraged sample diagnostics

Kronos demonstrates repeated stochastic sampling through `sample_count`, but averages outputs. MMFPS should do the opposite:

- preserve all 128 samples,
- score all 128,
- show best-of-K,
- measure diversity and containment.

This matches your current emergence dashboard direction.

### Borrow Later: Discrete behavior-code diagnostics

Use a tokenizer-like method not as the generator, but as a judge:

```text
real future paths -> behavior codes
generated futures -> behavior codes
compare code coverage
```

This could quantify whether generated futures cover the same behavioral modes as real futures.

## What MMFPS Should Not Borrow Yet

Do not immediately add:

- full tokenization,
- autoregressive replacement,
- discrete codebook loss,
- coarse/fine objective,
- Transformer LM objective,
- sample averaging.

MMFPS is currently in emergence validation. Kronos should inform the next research roadmap, not derail the current diffusion experiment.

## Suggested MMFPS Adaptation Roadmap

### Phase K0: No architecture change

Use Kronos as an idea source only. Continue validating MMFPS diffusion emergence and live dashboard behavior.

### Phase K1: Add analysis-only behavior codes

Train or derive a small offline path-shape codebook over real future trajectories. Use it only for evaluation:

- real code distribution,
- generated code distribution,
- best-of-128 code coverage,
- missed regimes.

No generator objective changes.

### Phase K2: Add explicit time/session conditioning

Inject session/time embeddings into MMFPS conditioning if not already strong enough:

- time of day,
- session phase,
- volatility regime,
- bar index within session.

This is low-risk and aligned with Kronos.

### Phase K3: Coarse-to-fine diffusion experiment

Only after MMFPS emergence is stable:

```text
coarse target: low-frequency path / trend / volatility envelope
fine target: residual returns
```

This would adapt Kronos’s hierarchy while preserving diffusion.

### Phase K4: Hybrid evaluator, not hybrid generator

Use a Kronos-like tokenizer as a discriminator/evaluator for financial path realism. Keep the MMFPS generator continuous.

## Commands Found In Kronos

Install:

```powershell
pip install -r requirements.txt
```

Run example prediction:

```powershell
python examples\prediction_example.py
```

Run BTC live test:

```powershell
python btc_live_test.py
```

Run Web UI:

```powershell
python webui\app.py
```

Open:

```text
http://127.0.0.1:7070
```

Prepare Qlib data:

```powershell
python finetune\qlib_data_preprocess.py
```

Fine-tune tokenizer with Qlib:

```powershell
torchrun --standalone --nproc_per_node=NUM_GPUS finetune\train_tokenizer.py
```

Fine-tune predictor with Qlib:

```powershell
torchrun --standalone --nproc_per_node=NUM_GPUS finetune\train_predictor.py
```

Run Qlib test/backtest:

```powershell
python finetune\qlib_test.py --device cuda:0
```

Run CSV sequential fine-tuning:

```powershell
cd finetune_csv
python train_sequential.py --config path\to\config.yaml
```

Run regression tests:

```powershell
pytest tests
```

## Final Judgment

Yes, this is a goldmine.

But the gold is architectural and representational, not copy-paste training code.

Kronos shows a strong alternative framing:

```text
market data as language
```

MMFPS is pursuing:

```text
market futures as stochastic diffusion manifold
```

Those are compatible research philosophies. The best future direction is likely not choosing one over the other, but letting Kronos influence MMFPS in controlled stages:

1. better temporal conditioning,
2. behavior-code diagnostics,
3. coarse/fine future structure,
4. eventually hierarchical diffusion.

The immediate recommendation is to keep MMFPS on its current emergence-validation path, then use Kronos ideas to design the next architecture phase once the current generator’s stochastic behavior is fully measured.
