#!/bin/bash
set -euo pipefail

# Reviewer comments:
# - quantify entropy stopping vs MSE-oracle stopping;
# - run a compact sensitivity grid for PT/entropy parameters.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATASET_DIR="${DATASET_DIR:-$HOME/projects/PTBenchmark/datasets}"
RUNNER="${RUNNER:-conda run -n ptbenchmark}"
PARTITION="${PARTITION:-cpu}"
LOG_DIR="${LOG_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results_partial}"
ANALYSIS_DIR="${ANALYSIS_DIR:-results/rev1}"

RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_ENTROPY="${RUN_ENTROPY:-1}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

DATASETS_STOPPING=(${DATASETS_STOPPING:-CBSD68 SEN2VENUS FMIDD_OH16230 FMIDD_PVD SIDD FORECAST MRI})
DATASETS_SENSITIVITY=(${DATASETS_SENSITIVITY:-CBSD68 MRI SIDD})
SENSITIVITY_METHODS=(${SENSITIVITY_METHODS:-perstree entropy})

# Compact default grid: enough to support the response without exploding runtime.
BASE_RELAX_SCALES="${BASE_RELAX_SCALES:-1e-6,1e-5,1e-4}"
GUIDED_RADII="${GUIDED_RADII:-1,2}"
EPSILON_SCALES="${EPSILON_SCALES:-1e-4,5e-4,1e-3}"
STOP_THRESHOLDS="${STOP_THRESHOLDS:-1e-5,1e-4,1e-3}"
MAX_ITER="${MAX_ITER:-100}"
MAX_IMAGES="${MAX_IMAGES:-}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$ANALYSIS_DIR"

ALL_JOB_IDS=()

join_by_colon() {
  local IFS=":"
  echo "$*"
}

submit_job() {
  local job_file="$1"
  local job_id
  job_id="$(sbatch --parsable "$job_file")"
  echo "Submitted $job_file as job $job_id"
  ALL_JOB_IDS+=("$job_id")
}

make_method_job() {
  local dataset="$1"
  local mode="$2"
  local method="$3"
  local output_stem="$4"
  local cpus="$5"
  local mem="$6"
  local time_limit="$7"
  local job_file="$8"

  cat > "$job_file" <<EOF
#!/bin/bash
#SBATCH --job-name=${dataset}_${method}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --time=${time_limit}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}

set -euo pipefail
cd "$REPO_DIR"

$RUNNER ptbenchmark test \\
  --dataset ${dataset} \\
  --mode ${mode} \\
  --method ${method} \\
  --datasets-dir "$DATASET_DIR" \\
  --output "${RESULTS_DIR}/${output_stem}.h5"
EOF
}

make_sensitivity_job() {
  local dataset="$1"
  local method="$2"
  local job_file="$3"
  local max_images_arg=""
  if [ -n "$MAX_IMAGES" ]; then
    max_images_arg="--max-images ${MAX_IMAGES}"
  fi

  cat > "$job_file" <<EOF
#!/bin/bash
#SBATCH --job-name=${dataset}_${method}_sens
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail
cd "$REPO_DIR"

$RUNNER ptbenchmark sensitivity \\
  --dataset ${dataset} \\
  --method ${method} \\
  --base-relax-scales "${BASE_RELAX_SCALES}" \\
  --guided-radii "${GUIDED_RADII}" \\
  --epsilon-scales "${EPSILON_SCALES}" \\
  --stop-thresholds "${STOP_THRESHOLDS}" \\
  --max-iter "${MAX_ITER}" \\
  --datasets-dir "$DATASET_DIR" \\
  --output "${RESULTS_DIR}/${dataset}_${method}_sensitivity.h5" \\
  --csv "${RESULTS_DIR}/${dataset}_${method}_sensitivity.csv" \\
  ${max_images_arg}
EOF
}

if [ "$RUN_ORACLE" = "1" ]; then
  for dataset in "${DATASETS_STOPPING[@]}"; do
    job="job_${dataset}_perstree_oracle.sbatch"
    make_method_job "$dataset" "method" "perstree" "${dataset}_perstree_oracle" "16" "64G" "1-00:00:00" "$job"
    submit_job "$job"
  done
fi

if [ "$RUN_ENTROPY" = "1" ]; then
  for dataset in "${DATASETS_STOPPING[@]}"; do
    job="job_${dataset}_entropy_stopping.sbatch"
    make_method_job "$dataset" "entropy" "entropy" "${dataset}_entropy_stopping" "4" "16G" "12:00:00" "$job"
    submit_job "$job"
  done
fi

if [ "$RUN_SENSITIVITY" = "1" ]; then
  for dataset in "${DATASETS_SENSITIVITY[@]}"; do
    for method in "${SENSITIVITY_METHODS[@]}"; do
      job="job_${dataset}_${method}_rev1_sensitivity.sbatch"
      make_sensitivity_job "$dataset" "$method" "$job"
      submit_job "$job"
    done
  done
fi

if [ "$RUN_ANALYSIS" = "1" ] && [ "${#ALL_JOB_IDS[@]}" -gt 0 ]; then
  dep_ids="$(join_by_colon "${ALL_JOB_IDS[@]}")"
  analysis_job="job_rev1_analysis.sbatch"
  cat > "$analysis_job" <<EOF
#!/bin/bash
#SBATCH --job-name=rev1_analysis
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --dependency=afterany:${dep_ids}

set -euo pipefail
cd "$REPO_DIR"

$RUNNER ptbenchmark merge \\
  --input-dir "${RESULTS_DIR}" \\
  --output "${ANALYSIS_DIR}/rev1_analysis.h5"

$RUNNER python ptbenchmark/results/analyze_oracle_vs_entropy.py \\
  --input "${ANALYSIS_DIR}/rev1_analysis.h5" \\
  --output-dir "${ANALYSIS_DIR}/entropy_stopping"
EOF
  job_id="$(sbatch --parsable "$analysis_job")"
  echo "Submitted $analysis_job as job $job_id with dependency afterany:$dep_ids"
fi

