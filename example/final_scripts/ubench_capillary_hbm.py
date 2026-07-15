#!/usr/bin/env python3
"""
Capillary (DSM) HBM-vs-theoretical microbenchmark, matching the PAT/FastTree
ubench workload (tree "1,{leaves}_{P},32", GQA 32/8, fp16 decode). Measures
Capillary decode latency and HBM bytes read, normalized to the theoretical-min
KV traffic, so a DSM line can be overlaid on the PAT/FastTree motivation plots.

Capillary = MultiLevelCascadeAttentionWrapper with tree-walk scheduling and the
DSM long-KV split (kvsplit_mode=True; mech1 fires for shared prefix >= 4096).

Two modes:
  --inner [--single]   set up + run ONE config (prints LATENCY_MS=... unless
                       --single, which does a single warmup-free launch for ncu)
  (default)            outer driver: sweep (leaves, prefix), shell to --inner
                       for latency and under ncu for HBM, write CSV.

Run (outer):
  cd .../flashinfer/example/final_scripts
  <myflashinfer_old_env python> ubench_capillary_hbm.py \
      --prefix-lens 8192 16384 32768 65536 131072 --leaves 64 128 256
"""
import argparse
import csv as csvmod
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
NCU = "/usr/local/cuda-12.9/bin/ncu"
NCU_TMPDIR = "/home/chris241/.tmp_ncu"
NUM_KV_HEADS = 8
GQA = 4          # -> 32 qo heads
HEAD_DIM = 128
DTYPE_BYTES = 2  # fp16
PAGE_SIZE = 16
LEAF_CTX = 32
MERGE_EVERY = 2
PACK_GQA_GLOBAL = False
# NOTE: do NOT match bare "fill" — flashinfer's kernel is "PreFILLWithKVCache".
SETUP_RE = re.compile(
    r"elementwise|distribution|curand|philox|randn|memset|arange|uniform|"
    r"catArray|indexSelect|copy_",
    re.IGNORECASE,
)


def theoretical_kv_bytes(P, leaves):
    unique = P + leaves * LEAF_CTX
    return unique * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES * 2  # K + V


# ---------------------------------------------------------------- inner ------
def build_and_run(nodes, contexts, kvsplit, warmup, reps, single, pack_gqa=False):
    import torch
    import flashinfer

    num_kv_heads, head_dim, page_size = NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE
    num_qo_heads = num_kv_heads * GQA
    num_levels = len(nodes)
    batch_size = nodes[-1]
    pages_per_node = [(c + page_size - 1) // page_size for c in contexts]
    total_pages = sum(nodes[l] * pages_per_node[l] for l in range(num_levels))
    total_queries = batch_size * num_levels

    kv_page_indices_list, kv_page_indptr_list, kv_last_page_len_list = [], [0], []
    page_offset = 0
    for level in range(num_levels):
        last = contexts[level] % page_size or page_size
        for _ in range(nodes[level]):
            kv_page_indices_list += list(range(page_offset, page_offset + pages_per_node[level]))
            kv_page_indptr_list.append(kv_page_indptr_list[-1] + pages_per_node[level])
            kv_last_page_len_list.append(last)
            page_offset += pages_per_node[level]
    dev = "cuda:0"
    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=dev)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=dev)
    kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=dev)

    qo_indptr_list = [0]
    for level in range(num_levels):
        seqs = batch_size // nodes[level]
        for _ in range(nodes[level]):
            qo_indptr_list.append(qo_indptr_list[-1] + seqs)
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=dev)

    kv_cache = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                           dtype=torch.float16, device=dev)
    q = torch.randn(total_queries, num_qo_heads, head_dim, dtype=torch.float16, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(1, ws, "NHD")
    wrapper.plan([qo_indptr], [kv_page_indptr], [kv_page_indices], [kv_last_page_len],
                 num_qo_heads, num_kv_heads, head_dim, page_size,
                 use_tree_walk_scheduling=True, kvsplit_mode=kvsplit, mech2_mode=False,
                 pack_gqa=pack_gqa)

    def run():
        return wrapper.run(q, kv_cache, tree_nodes=nodes, merge_every_n_levels=MERGE_EVERY)

    if single:
        run(); torch.cuda.synchronize(); return None

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record(); run(); ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return math.exp(sum(math.log(t) for t in times) / len(times))


# --------------------------------------------------------------- outer -------
def parse_ncu_bytes(stdout):
    lines = stdout.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith('"ID"') or ln.startswith("ID,") or '"Kernel Name"' in ln), None)
    if start is None:
        return None
    per = {}
    for row in csvmod.DictReader(lines[start:]):
        kn = row.get("Kernel Name")
        if not kn or "dram__bytes_read" not in row.get("Metric Name", ""):
            continue
        if SETUP_RE.search(kn):
            continue
        try:
            v = float(str(row.get("Metric Value", "")).replace(",", ""))
        except ValueError:
            continue
        mult = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9}.get(
            row.get("Metric Unit", "").strip().lower(), 1)
        per.setdefault(kn, []).append(v * mult)
    if not per:
        return None
    total = 0.0
    for vals in per.values():
        vals.sort(); total += vals[len(vals) // 2]
    return total


def inner_cmd(tree, kvsplit, warmup, reps, single):
    c = [PY, str(HERE / "ubench_capillary_hbm.py"), "--inner", "--tree", tree,
         "--warmup", str(warmup), "--reps", str(reps)]
    if kvsplit:
        c.append("--kvsplit")
    if single:
        c.append("--single")
    if PACK_GQA_GLOBAL:
        c.append("--pack")
    return c


def run_latency(tree, kvsplit, warmup, reps):
    p = subprocess.run(inner_cmd(tree, kvsplit, warmup, reps, False),
                       cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(f"[latency FAIL] {tree}\n{p.stderr[-1500:]}\n")
        return None
    m = re.search(r"LATENCY_MS=([0-9.]+)", p.stdout)
    return float(m.group(1)) if m else None


def run_hbm(tree, kvsplit, warmup, reps):
    env = dict(os.environ); env["TMPDIR"] = NCU_TMPDIR
    Path(NCU_TMPDIR).mkdir(parents=True, exist_ok=True)
    cmd = [NCU, "--metrics", "dram__bytes_read.sum", "--csv", "--target-processes", "all",
           "--"] + inner_cmd(tree, kvsplit, 0, 1, True)
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(f"[ncu FAIL] {tree}\n{p.stderr[-1500:]}\n")
        return None
    return parse_ncu_bytes(p.stdout)


def parse_tree(spec):
    nodes, contexts = spec.split("_")
    return list(map(int, nodes.split(","))), list(map(int, contexts.split(",")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--tree", type=str)
    ap.add_argument("--kvsplit", action="store_true", default=False)
    ap.add_argument("--pack", action="store_true", default=False)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--prefix-lens", type=int, nargs="+",
                    default=[8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--leaves", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--label", type=str, default="Capillary")
    ap.add_argument("--out", type=str, default="ubench_capillary_hbm")
    args = ap.parse_args()

    global PACK_GQA_GLOBAL
    PACK_GQA_GLOBAL = args.pack
    if args.inner:
        nodes, contexts = parse_tree(args.tree)
        lat = build_and_run(nodes, contexts, kvsplit=True, warmup=args.warmup,
                            reps=args.reps, single=args.single, pack_gqa=args.pack)
        if lat is not None:
            print(f"LATENCY_MS={lat:.6f}")
        return

    rows = []
    for leaves in args.leaves:
        for P in args.prefix_lens:
            tree = f"1,{leaves}_{P},{LEAF_CTX}"
            theo = theoretical_kv_bytes(P, leaves)
            print(f"[run] leaves={leaves:>4} P={P:>6} {tree}", flush=True)
            lat = run_latency(tree, True, args.warmup, args.reps)
            hbm = run_hbm(tree, True, args.warmup, args.reps)
            red = (hbm / theo) if hbm is not None else None
            print(f"      latency={lat}  hbm={hbm}  theo={theo:.3e}  redundancy={red}", flush=True)
            rows.append({"leaves": leaves, "prefix_len": P, "method": args.label,
                         "tree": tree, "latency_ms": lat, "hbm_bytes_read": hbm,
                         "theoretical_kv_bytes": theo, "redundancy": red})

    out = HERE / f"{args.out}.csv"
    with open(out, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
