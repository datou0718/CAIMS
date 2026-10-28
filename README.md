# CAIMS: Scalable Cross-Array In-Memory Search

This repository contains data
preparation, Dataset-Assisted
Scalable Search (DASS), one-time Online Result Count Adaptation (ORCA), functional
Local Aggregation and Selective Transfer (LAST), and accuracy evaluation.

## Install

Use Python **3.12 or newer**. The release is validated on Linux with Python 3.14;
CPU execution is supported, and CUDA can accelerate encoding and search.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints.txt .
caims --help
```

`requirements.txt` pins the direct dependencies. `constraints.txt` records the
validated transitive dependency versions. The installed package includes the
paper's recorded search parameters and works outside the checkout. Both
`caims` and `python -m caims` provide the same commands.

The standard PyTorch installation may include large CUDA libraries. For a CPU
installation, install its CPU wheel before the project:

```bash
python -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints.txt .
```

## Reproduce SciFact accuracy

Run the following from the checkout after installation. Downloads require
internet access. A GPU is optional; add `--devices cpu` to `encode` and
`--device cpu` to `last` to select CPU execution explicitly.

```bash
export CAIMS_DATA_ROOT="$PWD/data"

caims download --dataset scifact
caims encode --encoder BGE --dataset scifact
caims quantize --encoder BGE --dataset scifact --bits 3
caims paper-config --dataset scifact \
  --output "$CAIMS_DATA_ROOT/config/scifact-paper.json"
caims last --config "$CAIMS_DATA_ROOT/config/scifact-paper.json" \
  --output "$CAIMS_DATA_ROOT/scifact-matches.jsonl" --evaluate
```

The final command writes retrieved document IDs and prints an `Evaluation:` JSON
record. With the validated environment and recorded parameters, SciFact gives
**95.0% HitRate**, evaluated on 300 test queries. There are 1,109 encoded queries;
the mean result count across those queries is approximately **100.14**.
`--evaluate` calculates accuracy from retrieved results and evaluation labels;
it does not copy the recorded reference accuracy. Labels are not used by DASS,
ORCA, or search. Omit `--evaluate` for retrieval without relevance labels.

The output JSONL contains `query_id` and `document_ids`, in corpus order. Every
candidate tied at the selected score cutoff is retained; results are not
truncated to exactly 100 documents.

`CAIMS_DATA_ROOT` contains `dataset/`, `embedding/`, and `database/`. Without the
environment variable, it defaults to `data/` under the current working directory.
Set it before running commands or importing `caims`. Keep the same root across
commands. Datasets are `scifact`, `scidocs`, `quora`, `nq`, `nq-large`, and `tqa`;
the last two share the DPR Wikipedia corpus. Larger datasets require substantially
more storage and processing time than SciFact.

## Recorded parameters and reproducibility

[caims/data/reference_results.csv](caims/data/reference_results.csv) records the
original calibrated parameters and CAIMS accuracy for all six paper datasets.
These are archived paper statistics, not results of a newly sampled calibration.

| Dataset | T_filter | Final T_select | Recorded HitRate@100 |
| --- | ---: | ---: | ---: |
| BEIR-SciFact | -100 | 80 | 95.00% |
| BEIR-SciDocs | -92 | 84 | 88.80% |
| BEIR-Quora | -84 | 84 | 99.89% |
| BEIR-NQ | -100 | 78 | 92.00% |
| DPR-NQ (`nq-large`) | -84 | 74 | 88.20% |
| DPR-TQA (`tqa`) | -84 | 60 | 83.96% |

`T_filter` is the partition matching threshold. `T_select` is the final rank
after ORCA. The CSV also preserves the DASS rank, calibration mean count,
full-query mean count, and rounded accuracy printed in Table III. `paper-config`
loads these values without calibration.

All reference rows use:

- `BAAI/bge-base-en-v1.5`, pinned to revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
- Normalized float32 embeddings with 768 dimensions and maximum sequence length
  512. Corpus input is title + space + text; queries use raw text without a prefix.
- Three-bit per-vector Quantile Quantization, array width 24 (32 partitions),
  target count k = 100, and negative squared Euclidean similarity >= T_filter.

For identical quantized vectors, width, and final T_filter/T_select, retrieval
result sets are deterministic. Regenerated embeddings can differ near quantile
boundaries across software versions, devices, and numeric settings. The model
revision and dependency pins help control this variation. Existing embedding
files are reused; use a separate data root when changing the model or environment.
`MiniLM` is supported for exploration but is not the paper encoder and its model
revision is not pinned.

Fresh DASS calibration can choose a different pair because its corpus sample is
random. The original sample was unseeded and its indices are unavailable. Current
configurations save exact sampled rows. Do not expect every fresh seed to recover
the archived paper parameters or accuracy.

Accuracy uses positive BEIR test qrels or normalized token-sequence answer
matching for DPR. ORCA sampling and full-query mean counts use all encoded query
rows, preserving the original evaluation convention: SciFact has 1,109 encoded
queries / 300 test queries, and Quora has 15,000 / 10,000. Calibration `c` and
full-query mean counts describe different populations.

## Fresh DASS and ORCA calibration

After preparing vectors:

```bash
caims dass --encoder BGE --dataset scifact \
  --bits 3 --array-width 24 --k 100 --n-smp 1000 \
  --filter-count 8 --delta 0.10 --seed 42 \
  --output "$CAIMS_DATA_ROOT/config/scifact-dass.json"
caims orca --config "$CAIMS_DATA_ROOT/config/scifact-dass.json" \
  --n-online 300 --seed 42 \
  --output "$CAIMS_DATA_ROOT/config/scifact-orca.json"
caims last --config "$CAIMS_DATA_ROOT/config/scifact-orca.json" \
  --output "$CAIMS_DATA_ROOT/scifact-fresh-matches.jsonl" --evaluate
```

DASS excludes pseudo-query self matches, prefers stable pairs, and always falls
back to the closest unstable pair if none is stable. `--sample-indices file.npy`
reuses exact corpus row indices. ORCA keeps T_filter fixed and adapts T_select.
Both stages save their sampled rows in schema-2 configurations.

ORCA implements the one-time experimental adaptation in Section IV-B. Each call
calibrates without a runtime deviation trigger, and the proportional search
starts from k, preserving the original code. Here `delta` controls NeighborRefine
stability. The full runtime procedure in Algorithm 2 would measure c at the
current T_select, test deviation against delta, and update from that current rank
only when needed. That stateful controller is not implemented.

| Paper notation | Python / configuration | Meaning |
| --- | --- | --- |
| T_filter | `t_filter` | Uniform partition matching threshold |
| T_select | `t_select` | Selection rank, including all cutoff ties |
| k | `k` | Target average result count |
| c | `c` | Mean count on that stage's calibration queries |
| N_smp | `n_smp` | Number of corpus pseudo-queries |
| N_online | `n_online` | Number of online calibration queries |
| delta | `delta` | Relative count tolerance for NeighborRefine |
| array width | `array_width` | Dimensions per CAM partition |

DASS, ORCA, and LAST accept `--device`, `--query-batch-size`, and
`--document-batch-size`. Reduce batch sizes if device memory is limited. Encoding
accepts `--batch-size` and a comma-separated `--devices` list. `search` is an alias
for `last`.

## Implementation scope

LAST uses score histograms and a collection pass to reproduce the result sets of
descending-score broadcasts. NAND peripherals, the SSD pipeline, and the paper's
timing/energy model are not included. Software execution time is not a validation
of the modeled hardware throughput or energy.

The CPU baseline primitives use inner product on normalized full-precision
vectors, which gives cosine-similarity ranking. The generic IVF defaults are
4096 lists / 256 probes. Set `caims.baselines.IVF_NLIST` and `IVF_NPROBE` to the
Table I settings before calling `ivf_search`:

| Dataset | nlist | nprobe |
| --- | ---: | ---: |
| SciFact | 64 | 16 |
| SciDocs | 128 | 16 |
| Quora | 512 | 32 |
| BEIR-NQ | 1024 | 64 |
| DPR-NQ / DPR-TQA | 4096 | 256 |

The paper's CPU benchmarks use query batches of 16. Baseline helpers leave
batching to the caller. Experiment orchestration and generated results are not
part of this repository.

## Code map

| Module | Responsibility |
| --- | --- |
| `caims/dass.py` | Offline pseudo-query calibration and pair selection |
| `caims/orca.py` | One-time online result-count adaptation |
| `caims/selection.py` | Proportional updates, NeighborRefine, calibration result |
| `caims/cam.py` | Uniform threshold matching and partition-score aggregation |
| `caims/last.py` | Functional global selection with all cutoff ties |
| `caims/quantization.py` | Per-vector quantization and vector loading |
| `caims/baselines.py` | CPU exact and IVF retrieval primitives |
| `caims/metrics.py` | BEIR and DPR hit-rate evaluation |
| `caims/datasets.py` | Downloads and DPR preprocessing |
| `caims/embeddings.py` | Normalized embeddings with a pinned BGE revision |
| `caims/paths.py` | Portable data locations and shared DPR corpus mapping |
| `caims/runtime.py` | Device selection, transfer, and batching |
| `caims/reference.py` | Packaged paper parameters and accuracy metadata |
| `caims/configuration.py` | Configuration serialization and validation |
| `caims/__main__.py` | Command-line interface |

## Development and verification

```bash
python -m pip install -c constraints.txt -e .
python -m unittest discover -s tests -v
```

Tests use local fixtures without model or dataset downloads. GitHub Actions runs
CPU tests, builds source/wheel distributions, and checks the installed package
outside the checkout. To build locally:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

For bug reports, include the command, Python/dependency versions, device, and
configuration. Include sampled row indices when reporting calibration behavior.

## License and citation

The code and accompanying software documentation use the [MIT license](LICENSE).
Downloaded datasets, pretrained model weights, and third-party
dependencies retain their own licensing terms; the code license does not
relicense them. Dataset and model files are downloaded separately and excluded
from source distributions.

Software citation metadata is provided in [CITATION.cff](CITATION.cff).
