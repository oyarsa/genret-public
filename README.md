# GRAFT: facet-based generative paper retrieval

This repository contains the Python retrieval and training code, evaluation
scripts, and an optional web explorer. GRAFT generates facet-based document
identifiers with a fine-tuned language model and combines the retrieved papers
using reciprocal rank fusion (RRF), optionally weighted by a facet graph.

## Setup

Requirements: Python 3.14+, `uv`, and optionally `just`. Training and model
inference require a suitable NVIDIA GPU and CUDA environment. Access to
`meta-llama/Llama-3.2-1B-Instruct` requires accepting its model terms and
configuring Hugging Face authentication.

Run commands from the repository root:

```bash
# For training and GPU inference (x86-64 Linux).
uv sync --locked --extra cuda --extra training

# Alternatively, for CPU baselines and development.
# uv sync --locked --extra cpu --extra training

uv run --no-sync genret --help
```

The CPU and CUDA extras are mutually exclusive. The training commands below use
full fine-tuning on a GPU; installing the CPU extras does not reproduce that
hardware setup. Use `--no-sync` after setup to preserve the selected extras.

## Data and artifacts

This code-only distribution does **not** include datasets, extracted facets,
graphs, trained checkpoints, retrieval outputs, or relevance judgments. Exact
reproduction requires obtaining the original artifacts separately.

The main in-corpus results use **11,359 corpus papers and 500 test queries**.
Restore these paths relative to the repository root:

| Path                                                                   | Contents                                                                         |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `data/s2orc-nlp/splits-connected-11k/corpus_ids.txt`                   | Fixed corpus IDs                                                                 |
| `data/s2orc-nlp/splits-connected-11k/train.txt`                        | Training query IDs                                                               |
| `data/s2orc-nlp/splits-connected-11k/test.txt`                         | 500 held-out query IDs                                                           |
| `data/s2orc-nlp/splits-connected-11k/processed/facets.jsonl.zst`       | Original extracted facets for the corpus                                         |
| `data/s2orc-nlp/splits-connected-11k/processed/paper_graphs.jsonl.zst` | Paper metadata, including titles                                                 |
| `output/genret/codebook_training_pairs_11k.jsonl.zst`                  | Citation-derived training pairs and strict evaluation gold                       |
| `output/genret/facet_candidates_11k/`                                  | Four `facet_candidates_{problems,methods,results,contributions}.npz` graph files |

The pairs file contains multiple query splits. The training command explicitly
restricts source queries to `train.txt`; evaluation uses `test.txt`.

## Train the 11k GRAFT model

With the artifacts above available, the full recipe uses Llama-3.2-1B-Instruct,
all four facets, graph-weighted pair replication, a minimum of three target
views, and a top-10 graph-neighbor fallback:

```bash
export GRAFT_SPLITS=data/s2orc-nlp/splits-connected-11k
export GRAFT_PAIRS=output/genret/codebook_training_pairs_11k.jsonl.zst
export GRAFT_CANDIDATES=output/genret/facet_candidates_11k
export GRAFT_MODEL=meta-llama/Llama-3.2-1B-Instruct
export GRAFT_CHECKPOINT=output/genret/checkpoints/graft-reproduction

uv run --no-sync genret training glen train-bullets \
  --pairs "$GRAFT_PAIRS" \
  --facets "$GRAFT_SPLITS/processed/facets.jsonl.zst" \
  --paper-graphs "$GRAFT_SPLITS/processed/paper_graphs.jsonl.zst" \
  --train-split "$GRAFT_SPLITS/train.txt" \
  --output "$GRAFT_CHECKPOINT" --model "$GRAFT_MODEL" \
  --full-finetune --epochs 2 --batch-size 4 --grad-accum 4 \
  --lr 5e-5 --weight-decay 0.01 --max-seq-length 512 \
  --graph-sampling weighted --candidates-dir "$GRAFT_CANDIDATES" \
  --graph-sampling-max-copies 5 \
  --min-target-views 3 --min-views-graph-fallback 10
```

## Decode and score

Build the document identifier index with the same tokenizer as the model, or
restore the original index:

```bash
uv run --no-sync genret training glen build-docids \
  --facets "$GRAFT_SPLITS/processed/facets.jsonl.zst" \
  --model "$GRAFT_MODEL" --max-docid-tokens 12 \
  --output output/genret/glen_index_11k

mkdir -p output/genret/eval
uv run --no-sync genret training glen evaluate "$GRAFT_CHECKPOINT" \
  --facets "$GRAFT_SPLITS/processed/facets.jsonl.zst" \
  --glen-dir output/genret/glen_index_11k \
  --queries "$GRAFT_SPLITS/test.txt" \
  --per-bullet --beams 50 --max-docid-tokens 12 --top-n 50 \
  --output output/genret/eval/graft-reproduction.jsonl

uv run --no-sync python scripts/eval/score_glen_eval.py \
  output/genret/eval/graft-reproduction.jsonl \
  --gold strict --pairs "$GRAFT_PAIRS" \
  --graph-rrf --candidates-dir "$GRAFT_CANDIDATES" -k 5 10 20 50
```

The original checkpoint and evaluation settings above yielded strict-gold
**Recall@20 = 0.3261 and Recall@50 = 0.4533**. Strict gold consists of
citation-derived pairs; it is distinct from GPT relevance judgments. The
scorer averages recall over queries with nonempty gold. Omitting `--graph-rrf`
scores the decoder's ordinary fused ranking instead.

For basic training ablations, change these options and use a separate checkpoint
and decode output for each run:

| Variant                        | Options replacing the full recipe                                            |
|--------------------------------|------------------------------------------------------------------------------|
| Uniform training               | `--graph-sampling uniform --min-target-views 0 --min-views-graph-fallback 0` |
| Weighted sampling only         | `--min-target-views 0 --min-views-graph-fallback 0`                          |
| Minimum views without fallback | `--min-target-views 3 --min-views-graph-fallback 0`                          |

## Baselines and other evaluations

With the original baseline rankings and GPT judgment artifacts restored at the
paths configured in `scripts/eval/eval_baseline_table.py`, run:

```bash
uv run --no-sync python scripts/eval/eval_baseline_table.py 500q
```

This computes the GPT-labeled comparison table; its gold and recall definition
are different from the strict-gold scores above. The recorded strict-gold dense
graph teacher scores are Recall@20 = 0.357 and Recall@50 = 0.560.

To score a retrieval file against explicitly supplied GPT judgments:

```bash
uv run --no-sync python scripts/eval/score_glen_eval.py \
  output/genret/eval/graft-reproduction.jsonl \
  --gold gpt --gpt-labels output/eval_11k/gpt5mini_judgements_11k.jsonl \
  --graph-rrf --candidates-dir "$GRAFT_CANDIDATES" -k 5 10 20 50
```

Compare systems using the same query set and judgment pool. Changing the pool
changes the recall denominator. Out-of-corpus evaluation additionally requires
its original query IDs, extracted query facets, and citation gold; the unmatched
set has 351 queries with gold, while the year- and density-matched set has 139.
`scripts/eval/score_colbert_rebuttal.py` and
`scripts/eval/score_rebuttal_ablations.py` contain the corresponding artifact
paths and scoring logic, including paired bootstrap comparisons.

## Starting from raw papers

The acquisition and processing CLIs can create a new dataset. They require
Semantic Scholar access (configure `SEMANTIC_SCHOLAR_API_KEY`) and a configured
LLM provider for facet extraction. See `src/genret/config.py` and
`src/genret/llm/client.py` for environment settings.

```bash
uv run --no-sync genret pipeline all --help
uv run --no-sync genret pipeline splits --help
uv run --no-sync genret paper-processing --help
uv run --no-sync genret training build-codebook-data --help
```

The processing order is acquisition → fixed splits → parsing → facet extraction
→ embeddings → facet graphs → candidate graphs → edges → retrieval/training.
The dense teacher recipe uses MiniLM-L12-v2, dense weight 0.75, citation weight
0.25, n-gram weight 0, 100 candidates, 20 edges per facet, two BFS hops, ten
diversity slots, and RRF constant 60. A new data download, split, or facet
extraction is a new experiment and does not guarantee the original scores.

## Development

```bash
just lint  # Ruff fixes/formatting, Pyrefly, and unit tests
```

License: AGPL-3.0-or-later; see [LICENSE](LICENSE).
