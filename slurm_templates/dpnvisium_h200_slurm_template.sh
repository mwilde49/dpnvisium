#!/bin/bash
#SBATCH --job-name=dpnvisium-h200
#SBATCH --output=logs/dpnvisium_h200_%j.out
#SBATCH --error=logs/dpnvisium_h200_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200_nvl:1

module load apptainer

PROJECT_ROOT=/groups/tprice/pipelines
SCRATCH_ROOT=/scratch/juno/$USER
WORK_ROOT=/work/$USER

source "$PROJECT_ROOT/bin/lib/repro.sh"
source "$PROJECT_ROOT/bin/lib/provenance.sh"

CONTAINER=$PROJECT_ROOT/containers/dpnvisium/dpnvisium_v1.0.0.sif
PIPELINE_SCRIPT=$PROJECT_ROOT/containers/dpnvisium/dpnvisium.py

CONFIG=${1:-}
RUN_DIR=${2:-}
SCRATCH_OUTPUT_DIR=${3:-}

# --- Reproducibility capture (node, partition, resources, invocation log) ---
capture_juno_env "$RUN_DIR"
start_console_log "$RUN_DIR"
trap '_EC=$?; finalize_juno_env "$RUN_DIR" "$_EC"; generate_provenance_readme "$RUN_DIR" "dpnvisium-h200" "dpnvisium — Cell2Location Visium Deconvolution (GPU: H200, h200 partition)" "$_EC" "$CONTAINER" ""' EXIT

# --- Pre-flight checks ---

if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch dpnvisium_h200_slurm_template.sh <config.yaml> [run_dir] [scratch_output_dir]"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "Build it with: cd containers/dpnvisium/container && sudo apptainer build ../dpnvisium_v1.0.0.sif apptainer.def"
    echo "Then rsync dpnvisium_v1.0.0.sif to maw210003@juno.hpcre.utdallas.edu:$PROJECT_ROOT/containers/dpnvisium/"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found: $PIPELINE_SCRIPT"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# --- Software version capture ---
capture_software_versions "$RUN_DIR" "dpnvisium-h200" "$CONTAINER"

# --- Run pipeline ---

echo "====================================================================="
echo "  HYPERION COMPUTE — dpnvisium Pipeline Executing (GPU: H200)"
echo "====================================================================="
echo "NOTE: production config trains Cell2location full-batch (batch_size:"
echo "null) over ~40,600 spots across all 16 Visium samples concatenated."
echo "Targeting the h200 partition (26 nodes, 2x H200 NVL / 141GB VRAM each)"
echo "instead of h100 (3 nodes, 80GB VRAM) -- more VRAM headroom for this"
echo "step and a much deeper queue. See DPNVISIUM_HPC_GUIDE.md."

mkdir -p logs

run_logged "${RUN_DIR:+$RUN_DIR/invocation.log}" \
    apptainer exec \
    --nv \
    --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env MPLBACKEND=Agg \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    --bind $SCRATCH_ROOT:$SCRATCH_ROOT \
    --bind $WORK_ROOT:$WORK_ROOT \
    $CONTAINER \
    python $PIPELINE_SCRIPT --config $CONFIG
PIPELINE_EXIT=$?

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo "ERROR: Pipeline failed (exit $PIPELINE_EXIT). Skipping archive."
    exit $PIPELINE_EXIT
fi

# --- Stage-out: archive results from scratch to work ---

if [ -n "$RUN_DIR" ] && [ -n "$SCRATCH_OUTPUT_DIR" ]; then
    echo "[HYPERION] Data Relays Synchronizing — Archiving results to work"

    echo "Copying outputs: $SCRATCH_OUTPUT_DIR/ -> $RUN_DIR/outputs/"
    mkdir -p "$RUN_DIR/outputs"
    rsync -a --checksum "$SCRATCH_OUTPUT_DIR/" "$RUN_DIR/outputs/"

    echo "Verifying archive integrity..."
    OUTPUT_DIFF=$(rsync -a --checksum --dry-run "$SCRATCH_OUTPUT_DIR/" "$RUN_DIR/outputs/" 2>&1)
    if [ -z "$OUTPUT_DIFF" ]; then
        echo "[HYPERION] Data Relays Synchronized — Archive verification PASSED"
    else
        echo "[HYPERION] WARNING: Archive verification detected differences."
        echo "$OUTPUT_DIFF"
    fi
fi
