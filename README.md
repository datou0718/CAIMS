# CAIMS: Scalable Cross-Array In-Memory Search

Python simulation of CAIMS for scalable cross-array top-k similarity search.

## Installation

Requires Python 3.12 or later (tested with 3.14). CPU and CUDA are supported.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e .
```

## Usage

Prepare vectors, select search parameters with DASS, adapt them with ORCA, and
retrieve results with LAST. The example below uses SciFact:

```sh
export CAIMS_DATA_ROOT="$PWD/data"

caims download --dataset scifact
caims encode --encoder BGE --dataset scifact
caims quantize --encoder BGE --dataset scifact --bits 3

caims dass --encoder BGE --dataset scifact \
  --bits 3 --array-width 24 --k 100 --n-smp 1000 \
  --filter-count 8 --delta 0.10 --seed 42 \
  --output "$CAIMS_DATA_ROOT/config/scifact-dass.json"
caims orca --config "$CAIMS_DATA_ROOT/config/scifact-dass.json" \
  --n-online 300 --seed 42 \
  --output "$CAIMS_DATA_ROOT/config/scifact-orca.json"
caims last --config "$CAIMS_DATA_ROOT/config/scifact-orca.json" \
  --output "$CAIMS_DATA_ROOT/scifact-matches.jsonl"
```

DASS and ORCA save their parameters as JSON; LAST writes query and retrieved
document IDs as JSONL. Keep the same `CAIMS_DATA_ROOT` across commands.

To run on CPU, add `--devices cpu` to `encode` and `--device cpu` to `dass`, `orca`,
and `last`. Data preparation supports `scifact`, `scidocs`, `quora`, `nq`,
`nq-large`, and `tqa` and requires internet access for downloads.

## Algorithms

- **DASS (Dataset-Assisted Scalable Search):** Uses sampled candidate vectors as
  pseudo-queries to select the partition matching threshold `T_filter` and
  selection rank `T_select` offline.
- **ORCA (Online Result Count Adaptation):** Uses online queries to adapt
  `T_select` at a fixed `T_filter`, targeting an average of `k` results. Each call
  performs one adaptation.
- **LAST (Local Aggregation and Selective Transfer):** Implements global selection
  from aggregated partition match scores, retaining all candidates tied at the
  selection cutoff. The result count can therefore differ from exactly `k`.

## Search Parameters

| Parameter | Default | Command-line option |
|---|---|---|
| Encoder | BGE | `--encoder` |
| Quantization precision | 3 bits | `--bits` |
| CAM array width | 24 dimensions | `--array-width` |
| Target average result count | 100 | `--k` |
| DASS pseudo-queries | 1,000 | `--n-smp` |
| ORCA online queries | 300 | `--n-online` |

BGE uses `BAAI/bge-base-en-v1.5` to produce normalized 768-dimensional embeddings.
Vectors use per-vector Quantile Quantization and squared Euclidean distance.
The default array width gives 32 partitions per vector.

## Calibrated Parameters

Our parameters after DASS calibration and ORCA adaptation are included in
[reference_results.csv](caims/data/reference_results.csv). All use BGE, 3-bit
quantization, array width 24, and `k = 100`.

| Dataset | T_filter | T_select after DASS | T_select after ORCA |
|---|---:|---:|---:|
| `scifact` | -100 | 84 | 80 |
| `scidocs` | -92 | 84 | 84 |
| `quora` | -84 | 81 | 84 |
| `nq` | -100 | 80 | 78 |
| `nq-large` | -84 | 51 | 74 |
| `tqa` | -84 | 51 | 60 |

To use the final parameters directly after preparing vectors:

```sh
caims paper-config --dataset scifact \
  --output "$CAIMS_DATA_ROOT/config/scifact-ready.json"
caims last --config "$CAIMS_DATA_ROOT/config/scifact-ready.json" \
  --output "$CAIMS_DATA_ROOT/scifact-matches.jsonl"
```

## Citation

```bibtex
```

Licensed under the [MIT License](LICENSE).
