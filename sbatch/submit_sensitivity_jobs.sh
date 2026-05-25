#!/bin/bash
set -e

DATASETS=(CBSD68 SEN2VENUS FMIDD_OH16230 FMIDD_PVD SIDD FORECAST MRI)
METHODS=(perstree entropy)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$HOME/projects/PTBenchmark/datasets"
BATCH_SIZE="${BATCH_SIZE:-10}"

BASE_RELAX_SCALES="${BASE_RELAX_SCALES:-1e-6,1e-5,1e-4}"
GUIDED_RADII="${GUIDED_RADII:-1,2,3}"
EPSILON_SCALES="${EPSILON_SCALES:-1e-4,5e-4,1e-3}"
STOP_THRESHOLDS="${STOP_THRESHOLDS:-1e-5,1e-4,1e-3}"
MAX_IMAGES="${MAX_IMAGES:-}"

CURRENT_BATCH_IDS=()
PREVIOUS_BATCH_IDS=()

mkdir -p logs results_partial

join_by_colon() {
  local IFS=":"
  echo "$*"
}

submit_job() {
  local job_file="$1"
  local dependency_args=()
  local job_id

  if [ "${#PREVIOUS_BATCH_IDS[@]}" -gt 0 ]; then
    dependency_args=(--dependency="afterany:$(join_by_colon "${PREVIOUS_BATCH_IDS[@]}")")
  fi

  job_id="$(sbatch --parsable "${dependency_args[@]}" "$job_file")"
  echo "Submitted $job_file as job $job_id"
  CURRENT_BATCH_IDS+=("$job_id")

  if [ "${#CURRENT_BATCH_IDS[@]}" -ge "$BATCH_SIZE" ]; then
    PREVIOUS_BATCH_IDS=("${CURRENT_BATCH_IDS[@]}")
    CURRENT_BATCH_IDS=()
  fi
}

for DS in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    JOB="job_${DS}_${METHOD}_sensitivity.sbatch"
    MAX_IMAGES_ARG=""
    if [ -n "$MAX_IMAGES" ]; then
      MAX_IMAGES_ARG="--max-images ${MAX_IMAGES}"
    fi

    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --job-name=${DS}_${METHOD}_sens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd "$REPO_DIR"

micromamba run -n ptbenchmark ptbenchmark sensitivity \\
    --dataset ${DS} \\
    --method ${METHOD} \\
    --base-relax-scales "${BASE_RELAX_SCALES}" \\
    --guided-radii "${GUIDED_RADII}" \\
    --epsilon-scales "${EPSILON_SCALES}" \\
    --stop-thresholds "${STOP_THRESHOLDS}" \\
    --datasets-dir "${DATASET_DIR}" \\
    --output "results_partial/${DS}_${METHOD}_sensitivity.h5" \\
    --csv "results_partial/${DS}_${METHOD}_sensitivity.csv" \\
    ${MAX_IMAGES_ARG}
EOF
    submit_job "$JOB"
  done
done
