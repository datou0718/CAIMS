# CAIMS

Python simulation of **CAIMS: Scalable Cross-Array In-Memory Search**.

Includes DASS, ORCA, and LAST, with calibrated search parameters for six datasets.

## Installation

Requires Python 3.12 or later (tested with 3.14). Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -e .
```

## Quick start

```bash
export CAIMS_DATA_ROOT="$PWD/data"

caims download --dataset scifact
caims encode --encoder BGE --dataset scifact
caims quantize --encoder BGE --dataset scifact --bits 3
caims paper-config --dataset scifact \
  --output "$CAIMS_DATA_ROOT/config/scifact-ready.json"
caims last --config "$CAIMS_DATA_ROOT/config/scifact-ready.json" \
  --output "$CAIMS_DATA_ROOT/scifact-matches.jsonl"
```

The example prepares SciFact vectors, loads our calibrated parameters, and writes
query and retrieved document IDs to JSONL. Downloads require internet access.
To run on CPU, add `--devices cpu` to `encode` and `--device cpu` to `last`.
Keep the same `CAIMS_DATA_ROOT` across commands.

## Usage

After preparing vectors, call the algorithms directly to calibrate and search:

```python
from caims import dass, orca, last
from caims.quantization import load_vectors

cand_vec, qry_vec, _, _ = load_vectors("BGE", "scifact", bits=3)

params = dass.select_parameters(
    cand_vec, bits=3, array_width=24, k=100, n_smp=1000, device="cpu",
)
params = orca.adapt_t_select(
    cand_vec, qry_vec[:300], params.t_filter,
    array_width=24, k=100, device="cpu",
)
matches, _ = last.select_candidates(
    cand_vec, qry_vec, params.t_filter,
    array_width=24, t_select=params.t_select, device="cpu",
)
```

`cand_vec` and `qry_vec` are quantized NumPy arrays with one vector per row and
matching dimensions. `matches[i]` contains candidate row indices for query `i`.
Use `device="cuda:0"` for GPU execution.

### Calibrated parameters

Our parameters after DASS calibration and ORCA adaptation are included in
[reference_results.csv](caims/data/reference_results.csv). All use BGE
(`BAAI/bge-base-en-v1.5`), 3-bit Quantile Quantization, array width 24, and `k = 100`.

| Dataset | T_filter | T_select after DASS | T_select after ORCA |
| --- | ---: | ---: | ---: |
| `scifact` | -100 | 84 | 80 |
| `scidocs` | -92 | 84 | 84 |
| `quora` | -84 | 81 | 84 |
| `nq` | -100 | 80 | 78 |
| `nq-large` | -84 | 51 | 74 |
| `tqa` | -84 | 51 | 60 |

The quick start uses the final ORCA parameters. Replace `scifact` with another
dataset name to use its parameters.

## Algorithms

| Algorithm | Description |
| --- | --- |
| `DASS` | **Dataset-Assisted Scalable Search:** select `T_filter` and `T_select` offline using sampled candidates as pseudo-queries. |
| `ORCA` | **Online Result Count Adaptation:** adapt `T_select` at a fixed `T_filter` using online queries, targeting an average of `k` results. Each call performs one adaptation. |
| `LAST` | **Local Aggregation and Selective Transfer:** simulate global selection from aggregated partition match scores, retaining all candidates tied at the cutoff. |

## Citation

```bibtex
```

Licensed under the [MIT License](LICENSE).
