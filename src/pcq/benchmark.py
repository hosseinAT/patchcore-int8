"""FP32 vs INT8 PatchCore on MVTec AD.

    python -m pcq.benchmark --root /path/to/mvtec_anomaly_detection --out results

Variants (backbone / memory bank):
    fp32            FP32 backbone, FP32 memory bank            (reference = anomalib default)
    int8            INT8 backbone for memory bank AND queries
    int8_mixed      memory bank from FP32 features, queries from INT8 backbone
    int8_bank_fp16  `int8` + memory bank stored in FP16
    int8_bank_int8  `int8` + memory bank stored in INT8 (per-dimension scales)

`--smoke-test` runs the full pipeline on a tiny generated folder tree with a
randomly initialised backbone. It only checks that the code runs; its numbers
are meaningless and are never written into the README.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import CATEGORIES, find_root, load_category
from .pipeline import build_patchcore, embed, fit_memory_bank, score, time_module
from .quantize import compress_memory_bank, quantize_backbone, relative_feature_error, state_dict_bytes

VARIANTS = ("fp32", "int8", "int8_mixed", "int8_bank_fp16", "int8_bank_int8")
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def run_category(root, category, args, compute_device) -> dict:
    t_start = time.time()
    data = load_category(root, category, args.size, args.max_train, args.max_test)
    model = build_patchcore(pretrained=not args.smoke_test).to(compute_device)  # blur kernel etc. must sit on the compute device
    fp32_bb = model.feature_extractor.feature_extractor.eval()
    calib = data["train"][: args.calib_images]
    int8_bb = quantize_backbone(fp32_bb, calib)
    fp32_dev = compute_device  # FP32 backbone may use the GPU for speed; INT8 kernels are CPU-only
    fp32_bb.to(fp32_dev)

    banks = {
        "fp32": fit_memory_bank(model, fp32_bb, data["train"], fp32_dev, compute_device, args.coreset),
        "int8": fit_memory_bank(model, int8_bb, data["train"], "cpu", compute_device, args.coreset),
    }
    banks["int8_mixed"] = banks["fp32"]
    sizes = {"fp32": banks["fp32"].numel() * 4}
    for mode in ("fp16", "int8"):
        model.memory_bank = banks["int8"].clone()
        sizes[f"int8_bank_{mode}"] = compress_memory_bank(model, mode)
        banks[f"int8_bank_{mode}"] = model.memory_bank.clone()
    sizes["int8"] = sizes["int8_mixed"] = banks["int8"].numel() * 4

    which_backbone = {"fp32": "fp32", "int8": "int8", "int8_mixed": "int8", "int8_bank_fp16": "int8", "int8_bank_int8": "int8"}
    preds = {v: [] for v in VARIANTS}
    maps = {v: [] for v in VARIANTS}
    feat_err = []
    image_size = data["test"].shape[-2:]
    for i in range(0, len(data["test"]), 8):
        x = data["test"][i : i + 8]
        e32, shape = next(embed(model, fp32_bb, x, fp32_dev, compute_device, batch=len(x)))
        e8, _ = next(embed(model, int8_bb, x, "cpu", compute_device, batch=len(x)))
        feat_err.append(relative_feature_error(e32, e8))
        for v in VARIANTS:
            p, m = score(model, banks[v], e32 if which_backbone[v] == "fp32" else e8, shape, image_size)
            preds[v].append(p)
            maps[v].append(m)

    labels = data["labels"].numpy()
    masks = data["masks"].numpy().ravel().astype(np.uint8)
    res = {"category": category, "n_train": len(data["train"]), "n_test": len(labels), "variants": {}}
    for v in VARIANTS:
        s = torch.cat(preds[v]).numpy()
        m = torch.cat(maps[v]).numpy().ravel()
        res["variants"][v] = {
            "image_auroc": float(roc_auc_score(labels, s)) if 0 < labels.sum() < len(labels) else None,
            "pixel_auroc": float(roc_auc_score(masks, m)) if 0 < masks.sum() < len(masks) else None,
            "memory_bank_bytes": int(sizes[v]),
            "memory_bank_vectors": int(banks[v].shape[0]),
        }
        res["variants"][v]["_scores"] = s.tolist()
    s32 = np.array(res["variants"]["fp32"]["_scores"])
    for v in VARIANTS:
        sv = np.array(res["variants"][v]["_scores"])
        res["variants"][v]["score_spearman_vs_fp32"] = float(np.corrcoef(np.argsort(np.argsort(s32)), np.argsort(np.argsort(sv)))[0, 1])
    res["feature_rel_error_int8"] = float(statistics.mean(feat_err))

    # CPU latency micro-benchmark (batch 1)
    x1 = data["test"][:1]
    model.to("cpu")  # latency is measured on CPU: move the whole model, not only the backbone
    fp32_cpu = fp32_bb
    res["latency_ms_cpu"] = {"backbone_fp32": time_module(fp32_cpu, x1, runs=args.timing_runs), "backbone_int8": time_module(int8_bb, x1, runs=args.timing_runs)}
    e8, shape = next(embed(model, int8_bb, x1, "cpu", "cpu", batch=1))
    res["latency_ms_cpu"]["knn_fp32_bank"] = time_module(lambda _: score(model, banks["fp32"].cpu(), e8, shape, image_size), x1, runs=args.timing_runs)
    res["model_bytes"] = {"backbone_fp32": state_dict_bytes(fp32_cpu), "backbone_int8": state_dict_bytes(int8_bb)}
    res["seconds"] = time.time() - t_start
    return res


def summarise(results: list[dict]) -> dict:
    out = {"categories": [r["category"] for r in results], "mean": {}}
    for v in VARIANTS:
        for key in ("image_auroc", "pixel_auroc"):
            vals = [r["variants"][v][key] for r in results if r["variants"][v][key] is not None]
            out["mean"].setdefault(v, {})[key] = statistics.mean(vals) if vals else None
    lat = lambda k: statistics.median(r["latency_ms_cpu"][k] for r in results)
    out["latency_ms_cpu"] = {k: lat(k) for k in results[0]["latency_ms_cpu"]}
    out["model_bytes"] = results[0]["model_bytes"]
    out["feature_rel_error_int8"] = statistics.mean(r["feature_rel_error_int8"] for r in results)
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def markdown(results: list[dict], summary: dict, env: dict) -> str:
    L = [f"*Generated by `python -m pcq.benchmark` on {env['date']} · {env['device']} · torch {env['torch']} · anomalib {env['anomalib']}*\n",
         "#### Image-level AUROC (%) per category\n",
         "| Category | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    for r in results:
        L.append(f"| {r['category']} | " + " | ".join(pct(r["variants"][v]["image_auroc"]) for v in VARIANTS) + " |")
    L.append("| **mean** | " + " | ".join(f"**{pct(summary['mean'][v]['image_auroc'])}**" for v in VARIANTS) + " |")
    L += ["\n#### Pixel-level AUROC (%) per category\n", "| Category | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    for r in results:
        L.append(f"| {r['category']} | " + " | ".join(pct(r["variants"][v]["pixel_auroc"]) for v in VARIANTS) + " |")
    L.append("| **mean** | " + " | ".join(f"**{pct(summary['mean'][v]['pixel_auroc'])}**" for v in VARIANTS) + " |")
    lat, mb = summary["latency_ms_cpu"], summary["model_bytes"]
    bank = {v: statistics.mean(r["variants"][v]["memory_bank_bytes"] for r in results) for v in VARIANTS}
    L += ["\n#### Cost (CPU, batch 1, median over categories)\n", "| | FP32 | INT8 | Ratio |", "|---|---:|---:|---:|",
          f"| Backbone latency [ms] | {lat['backbone_fp32']:.0f} | {lat['backbone_int8']:.0f} | {lat['backbone_fp32'] / lat['backbone_int8']:.2f}× faster |",
          f"| Backbone size [MB] | {mb['backbone_fp32'] / 1e6:.0f} | {mb['backbone_int8'] / 1e6:.0f} | {mb['backbone_fp32'] / mb['backbone_int8']:.2f}× smaller |",
          f"| Memory bank [MB] (fp32 → int8 storage) | {bank['fp32'] / 1e6:.1f} | {bank['int8_bank_int8'] / 1e6:.1f} | {bank['fp32'] / bank['int8_bank_int8']:.2f}× smaller |",
          f"| Nearest-neighbour search [ms] | {lat['knn_fp32_bank']:.0f} | – | unchanged (search runs in float) |",
          f"\nMean relative L2 error of INT8 vs FP32 patch embeddings: **{100 * summary['feature_rel_error_int8']:.1f} %**."]
    return "\n".join(L) + "\n"


def update_readme(readme: Path, md: str):
    text = readme.read_text()
    new = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), README_START + "\n" + md + README_END, text, flags=re.S)
    readme.write_text(new)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=["datasets", "/kaggle/input", "/content/datasets"])
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--calib-images", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--timing-runs", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--smoke-test", action="store_true", help="random weights + tiny fake data; numbers are meaningless")
    args = ap.parse_args(argv)

    if args.smoke_test:
        from .smoke import make_fake_mvtec

        root = make_fake_mvtec(Path("/tmp/pcq_fake_mvtec"), args.categories[:2])
        args.categories = args.categories[:2]
        args.size, args.calib_images, args.timing_runs = 64, 4, 2
    else:
        root = find_root(args.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"MVTec root: {root} · device: {device} · threads: {torch.get_num_threads()}", flush=True)

    results = []
    for cat in args.categories:
        r = run_category(root, cat, args, device)
        results.append(r)
        (args.out / f"{cat}.json").write_text(json.dumps(r, indent=1))
        v = r["variants"]
        print(f"{cat:12s} image AUROC fp32 {pct(v['fp32']['image_auroc'])} | int8 {pct(v['int8']['image_auroc'])} | "
              f"int8+bank8 {pct(v['int8_bank_int8']['image_auroc'])}   ({r['seconds']:.0f}s)", flush=True)

    import anomalib

    env = {"date": time.strftime("%Y-%m-%d"), "device": f"{device} ({platform.processor() or platform.machine()})",
           "torch": torch.__version__, "anomalib": anomalib.__version__, "smoke_test": args.smoke_test}
    summary = summarise(results)
    md = markdown(results, summary, env)
    (args.out / "summary.json").write_text(json.dumps({"env": env, "args": {k: str(v) for k, v in vars(args).items()}, "summary": summary}, indent=1))
    (args.out / "results.md").write_text(md)
    if args.smoke_test:
        print("SMOKE TEST – numbers are meaningless, README not updated.")
    elif args.readme.exists():
        update_readme(args.readme, md)
        print(f"README updated: {args.readme}")
    print(md)


if __name__ == "__main__":
    main()
