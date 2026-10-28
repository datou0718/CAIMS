"""CAIMS data preparation, DASS, ORCA, and LAST command-line interface."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .configuration import SCHEMA_VERSION, read_configuration, write_configuration
from .paths import DATASETS


def add_dataset_arguments(parser):
    parser.add_argument("--encoder", choices=["BGE", "MiniLM"], default="BGE")
    parser.add_argument("--dataset", choices=DATASETS, required=True)


def add_compute_arguments(parser):
    parser.add_argument("--device", help="For example cpu or cuda:0; selected automatically if omitted")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--document-batch-size", type=int, default=65536)


def compute_options(args):
    return dict(device=args.device, query_batch_size=args.query_batch_size,
                document_batch_size=args.document_batch_size)


def build_parser():
    parser = argparse.ArgumentParser(description="CAIMS: DASS, ORCA, and LAST")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download BEIR or shared DPR data")
    download.add_argument("--dataset", choices=DATASETS, required=True)

    encode = commands.add_parser("encode", help="Encode normalized candidate/query vectors")
    add_dataset_arguments(encode)
    encode.add_argument("--batch-size", type=int, default=256)
    encode.add_argument("--corpus-chunk-size", type=int, default=16384)
    encode.add_argument("--devices", help="Comma-separated devices; selected automatically if omitted")

    quantize = commands.add_parser("quantize", help="Apply per-vector Quantile Quantization")
    add_dataset_arguments(quantize)
    quantize.add_argument("--bits", type=int, default=3)

    dass = commands.add_parser("dass", help="Offline selection of T_filter and T_select (Algorithm 1)")
    add_dataset_arguments(dass)
    add_compute_arguments(dass)
    dass.add_argument("--bits", type=int, default=3)
    dass.add_argument("--array-width", type=int, default=24)
    dass.add_argument("--k", type=int, default=100, help="Target average result count k")
    dass.add_argument("--n-smp", type=int, default=1000, help="Number of corpus pseudo-queries N_smp")
    dass.add_argument("--seed", type=int, default=42)
    dass.add_argument("--sample-indices", type=Path, help="Existing pseudo-query .npy indices")
    dass.add_argument("--delta", type=float, default=.10, help="Relative count tolerance delta")
    dass.add_argument("--filter-count", type=int, default=8, help="Number of candidate T_filter values")
    dass.add_argument("--output", type=Path, required=True)

    orca = commands.add_parser("orca", help="Adapt T_select with online queries at fixed T_filter")
    add_compute_arguments(orca)
    orca.add_argument("--config", type=Path, required=True)
    orca.add_argument("--n-online", type=int, default=300, help="Number of online queries N_online")
    orca.add_argument("--seed", type=int, default=42)
    orca.add_argument("--delta", type=float, help="Override the saved count tolerance")
    orca.add_argument("--output", type=Path, required=True)

    last = commands.add_parser("last", aliases=["search"], help="Run LAST's tie-inclusive global selection")
    add_compute_arguments(last)
    last.add_argument("--config", type=Path, required=True)
    last.add_argument("--output", type=Path, required=True, help="Write query/document IDs as JSONL")
    last.add_argument("--evaluate", action="store_true",
                      help="Report measured HitRate and mean count using the dataset's evaluation labels")

    paper = commands.add_parser("paper-config", help="Load recorded paper parameters without recalibration")
    paper.add_argument("--dataset", choices=DATASETS, required=True)
    paper.add_argument("--output", type=Path, required=True)
    return parser


def dass_command(args):
    import numpy as np
    from .dass import sample_pseudo_query_indices, select_parameters
    from .quantization import load_vectors

    cand_vec, _, _, _ = load_vectors(args.encoder, args.dataset, args.bits)
    sample_indices = (np.load(args.sample_indices) if args.sample_indices else
                      sample_pseudo_query_indices(len(cand_vec), args.n_smp, args.seed))
    selected = select_parameters(
        cand_vec, bits=args.bits, array_width=args.array_width, k=args.k,
        query_indices=sample_indices, delta=args.delta, filter_count=args.filter_count,
        **compute_options(args),
    )
    configuration = {
        "schema_version": SCHEMA_VERSION, "source": "calibration",
        "encoder": args.encoder, "dataset": args.dataset, "bits": args.bits,
        "array_width": args.array_width, "k": args.k, **asdict(selected),
        "dass": asdict(selected), "orca": None,
        "seed": args.seed, "delta": args.delta, "filter_count": args.filter_count,
        "n_smp": len(sample_indices), "n_online": 0,
        "pseudo_query_indices": sample_indices.tolist(), "online_query_indices": [],
    }
    write_configuration(args.output, configuration)
    print(f"Saved {args.output}: T_filter={selected.t_filter:g}, "
          f"T_select={selected.t_select}, c={selected.c:.2f}, stable={selected.stable}")


def orca_command(args):
    import numpy as np
    from .orca import adapt_t_select
    from .quantization import load_vectors

    configuration = read_configuration(args.config)
    cand_vec, qry_vec, _, _ = load_vectors(
        configuration["encoder"], configuration["dataset"], configuration["bits"],
    )
    if not 0 < args.n_online <= len(qry_vec):
        raise ValueError(f"n-online must be between 1 and {len(qry_vec)}")
    indices = np.random.default_rng(args.seed).permutation(len(qry_vec))[:args.n_online]
    delta = args.delta if args.delta is not None else configuration.get("delta", .10)
    selected = adapt_t_select(
        cand_vec, qry_vec[indices], configuration["t_filter"],
        array_width=configuration["array_width"], k=configuration["k"], delta=delta,
        **compute_options(args),
    )
    configuration.update({
        **asdict(selected), "source": "calibration", "orca": asdict(selected),
        "online_seed": args.seed, "delta": delta, "n_online": len(indices),
        "online_query_indices": indices.tolist(),
    })
    write_configuration(args.output, configuration)
    print(f"Saved {args.output}: T_filter={selected.t_filter:g}, "
          f"T_select={selected.t_select}, c={selected.c:.2f}, stable={selected.stable}")


def last_command(args):
    from .last import select_candidates
    from .quantization import load_vectors
    from .runtime import select_device

    configuration = read_configuration(args.config)
    cand_vec, qry_vec, document_ids, query_ids = load_vectors(
        configuration["encoder"], configuration["dataset"], configuration["bits"],
    )
    results, _ = select_candidates(
        cand_vec, qry_vec, configuration["t_filter"], configuration["array_width"],
        configuration["t_select"], args.device or select_device(max_memory_used_percent=99),
        args.query_batch_size, args.document_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for query_id, indices in zip(query_ids, results):
            ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                   for value in document_ids[indices]]
            destination.write(json.dumps({"query_id": query_id, "document_ids": ids}) + "\n")
    print(f"Wrote results for {len(query_ids)} queries to {args.output}")
    if args.evaluate:
        from .metrics import (
            ANSWER_EVALUATION_DATASETS, answer_hit_rate_at_k, hit_rate_at_k,
            load_answers, load_qrels,
        )

        dataset = configuration["dataset"]
        if dataset in ANSWER_EVALUATION_DATASETS:
            answers = load_answers(dataset)
            evaluated_queries = sum(bool(answers.get(query_id)) for query_id in query_ids)
            accuracy = answer_hit_rate_at_k(results, query_ids, dataset, len(document_ids))
        else:
            qrels = load_qrels(dataset)
            evaluated_queries = sum(
                any(value > 0 for value in qrels.get(query_id, {}).values())
                for query_id in query_ids
            )
            accuracy = hit_rate_at_k(results, query_ids, document_ids, qrels)
        if not evaluated_queries:
            raise ValueError("No queries have evaluation labels; accuracy cannot be measured")
        summary = {
            "dataset": dataset, "k": configuration["k"],
            "t_filter": configuration["t_filter"], "t_select": configuration["t_select"],
            "hit_rate": accuracy, "hit_rate_percent": 100 * accuracy,
            "evaluated_queries": evaluated_queries, "encoded_queries": len(query_ids),
            "mean_result_count_all_queries": sum(map(len, results)) / len(results),
        }
        print("Evaluation: " + json.dumps(summary, sort_keys=True))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "download":
        from .datasets import ensure_dataset
        print(ensure_dataset(args.dataset))
    elif args.command == "encode":
        from .embeddings import encode_dataset
        encode_dataset(args.encoder, args.dataset, args.batch_size, args.corpus_chunk_size,
                       args.devices.split(",") if args.devices else None)
    elif args.command == "quantize":
        from .quantization import load_vectors
        cand_vec, qry_vec, _, _ = load_vectors(args.encoder, args.dataset, args.bits)
        print(f"Prepared cand_vec={cand_vec.shape}, qry_vec={qry_vec.shape}, bits={args.bits}")
    elif args.command == "dass":
        dass_command(args)
    elif args.command == "orca":
        orca_command(args)
    elif args.command in {"last", "search"}:
        last_command(args)
    elif args.command == "paper-config":
        from .reference import paper_configuration
        configuration = paper_configuration(args.dataset)
        write_configuration(args.output, configuration)
        print(f"Saved recorded paper parameters to {args.output}: "
              f"T_filter={configuration['t_filter']:g}, T_select={configuration['t_select']}")


if __name__ == "__main__":
    main()
