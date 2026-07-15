#!/bin/bash
# =============================================================================
# Reproduce capillary_vs_baselines.pdf end-to-end on a Hopper GPU (H100/H200).
#
# The figure is a PAT-normalized latency comparison (higher = faster) of four
# external baselines (PAT, FastTree, FlashAttention3, Cascade) against six
# Capillary variants (kvsplit cluster 2/4/8, pack_gqa, kvsplit+pack cluster
# 2/4). This script re-measures ALL of it on the current hardware and rebuilds
# the plot, so on an H200 you get an H200 figure (the committed one is H100).
#
# H200 note: H200 is Hopper (SM90), identical to H100 for these SM90 kernels;
# TORCH_CUDA_ARCH_LIST=9.0 and the kClusterSize logic are unchanged. No code
# changes are needed versus H100.
#
# Runtime: ~3 hours (72 baseline kernel runs + 6 Capillary sweeps + 5 flashinfer
# rebuilds at different cluster sizes). Use --skip-baselines / --skip-capillary
# to rerun a stage.
#
# Prerequisites (paths below; edit if your layout differs):
#   - PAT baseline venv + benchmark:  $PAT_VENV, $PAT_BENCH
#   - Capillary (this repo) venv:     $CAP_VENV, built via $FLASHINFER/compile.sh
#   - Both are the same as used for the committed figure.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./reproduce_capillary_vs_baselines.sh [OUTDIR]
#   CUDA_VISIBLE_DEVICES=0 ./reproduce_capillary_vs_baselines.sh --skip-baselines OUTDIR
# =============================================================================
set -euo pipefail

# ---- paths (edit for your environment) --------------------------------------
FLASHINFER=/home/chris241/myflashinfer_old/flashinfer
CAP_VENV=/home/chris241/myflashinfer_old/myflashinfer_old_env/bin/python
CAP_ENV_ACT=/home/chris241/myflashinfer_old/myflashinfer_old_env/bin/activate
PAT_ROOT=/home/chris241/myflashinfer_old/baselines/PAT_workspace
PAT_VENV=$PAT_ROOT/pat_venv/bin/python
PAT_BENCH=$PAT_ROOT/PAT/benchmark
FS_DIR=$FLASHINFER/example/final_scripts
CLUSTER_HDR=$FLASHINFER/include/flashinfer/attention/hopper/cluster_config.cuh

SKIP_BASELINES=0; SKIP_CAP=0; OUTDIR=""
for a in "$@"; do
  case "$a" in
    --skip-baselines) SKIP_BASELINES=1 ;;
    --skip-capillary) SKIP_CAP=1 ;;
    *) OUTDIR="$a" ;;
  esac
done
OUTDIR="${OUTDIR:-$FS_DIR/repro_$(hostname -s)}"
mkdir -p "$OUTDIR"
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer-repro-$USER}"
echo "Output dir: $OUTDIR"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Paper tree suite (24 trees) and head configs (3 panels).
TREES=(1,64_16384,32 1,64_32768,32 1,64_65536,32 1,64_131072,32 \
       1,64_16384,64 1,64_32768,64 1,64_65536,64 1,64_131072,64 \
       1,64_16384,128 1,64_32768,128 1,64_65536,128 1,64_131072,128 \
       1,128_16384,32 1,128_32768,32 1,128_65536,32 1,128_131072,32 \
       1,128_16384,64 1,128_32768,64 1,128_65536,64 1,128_131072,64 \
       1,128_16384,128 1,128_32768,128 1,128_65536,128 1,128_131072,128)
HEADS=("32 32" "32 8" "16 8")   # (nheads_q nheads_kv) -> GQA 1, 4, 2
BLOCK=32
PAT_JSON="$OUTDIR/baselines.json"

# ---- Stage 1: external baselines (PAT/FastTree/FA3/Cascade), pat_venv --------
if [[ $SKIP_BASELINES -eq 0 ]]; then
  echo "== Stage 1/3: baseline kernels (pat,cascade,FastTree,fa) =="
  rm -f "$PAT_JSON"
  for hc in "${HEADS[@]}"; do
    set -- $hc; HQ=$1; HKV=$2
    for t in "${TREES[@]}"; do
      echo "  [baseline] heads=$HQ/$HKV tree=$t"
      ( cd "$PAT_BENCH" && WARMUP=20 N_REPEATS=50 "$PAT_VENV" benchmark_kernel.py \
          --tree "$t" --nheads_q "$HQ" --nheads_kv "$HKV" \
          --baselines "pat,cascade,FastTree,fa" --block_size "$BLOCK" \
          --output_file "$PAT_JSON" >/dev/null 2>&1 ) || echo "    (skipped: OOM/limit)"
    done
  done
  echo "  wrote $PAT_JSON"
fi

# ---- Stage 2: Capillary variants, rebuild per cluster size ------------------
rebuild_cluster() {  # $1 = cluster size
  sed -i "s/constexpr int kClusterSize = [0-9]*;/constexpr int kClusterSize = $1;/" "$CLUSTER_HDR"
  ( cd "$FLASHINFER" && source env.sh && source "$CAP_ENV_ACT" && bash compile.sh ) >/dev/null 2>&1
}
cap_run() {  # $1 = out csv basename, rest = extra flags (--kvsplit_mode/--pack_gqa)
  local out="$OUTDIR/$1"; shift
  local args=(); for t in "${TREES[@]}"; do args+=(--tree "$t"); done
  ( cd "$FS_DIR" && "$CAP_VENV" bench_mech2_GQA_only.py "${args[@]}" \
      --num_kv_heads 8 --warmup_iters 20 --bench_iters 50 \
      --output_csv "$out" "$@" ) >/dev/null 2>&1
  echo "  wrote $out"
}

if [[ $SKIP_CAP -eq 0 ]]; then
  echo "== Stage 2/3: Capillary sweeps (rebuild per cluster size) =="
  for C in 2 4 8; do
    echo "  -- cluster $C --"; rebuild_cluster "$C"
    cap_run "bench_mech2_gqa_sweep_kvsplit_cluster${C}.csv" --kvsplit_mode
    if [[ "$C" == "4" ]]; then
      cap_run "bench_pack_on.csv" --pack_gqa                    # pack only
    fi
    if [[ "$C" == "2" || "$C" == "4" ]]; then
      cap_run "bench_pack_kvsplit_cluster${C}.csv" --kvsplit_mode --pack_gqa
    fi
  done
  echo "  restoring kClusterSize=4"; rebuild_cluster 4
fi

# ---- Stage 3: plot ----------------------------------------------------------
echo "== Stage 3/3: plot =="
"$CAP_VENV" "$FS_DIR/reproduce_capillary_vs_baselines.py" \
    --pat-json "$PAT_JSON" --cap-dir "$OUTDIR" \
    --out "$OUTDIR/capillary_vs_baselines"
echo "Done. Figure: $OUTDIR/capillary_vs_baselines.pdf"
