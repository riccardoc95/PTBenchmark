#!/bin/bash
set -e

DATASETS=(CBSD68 SEN2VENUS FMIDD_OH16230 FMIDD_PVD SIDD FORECAST MRI)
SUPERVISED=(unet dncnn)
UNSUPERVISED=(perstree perstree_cut peronamalik bm3d nlmeans gaussian median wavelet)
DISTANCES=(RD RF)

DATASET_DIR="$HOME/projects/PTBenchmark/datasets"

mkdir -p logs results_partial

# === SUPERVISED ===
for DS in "${DATASETS[@]}"; do
  for M in "${SUPERVISED[@]}"; do
    JOB="job_${DS}_${M}.sbatch"
    sed \
      -e "s/{JOBNAME}/${DS}_${M}/g" \
      -e "s/{DATASET}/${DS}/g" \
      -e "s/{MODE}/method/g" \
      -e "s/{METHOD}/${M}/g" \
      -e "s|{EXTRA_ARGS}|--method ${M} --epochs 20 --device cpu --datasets-dir ${DATASET_DIR}|g" \
      -e "s/{OUTPUT}/${DS}_${M}/g" \
      -e "s/{CPUS}/16/g" \
      -e "s/{MEM}/64G/g" \
      job_template.sbatch > "$JOB"
    sbatch "$JOB"
  done
done

# === UNSUPERVISED ===
for DS in "${DATASETS[@]}"; do
  for M in "${UNSUPERVISED[@]}"; do
    JOB="job_${DS}_${M}.sbatch"
    sed \
      -e "s/{JOBNAME}/${DS}_${M}/g" \
      -e "s/{DATASET}/${DS}/g" \
      -e "s/{MODE}/method/g" \
      -e "s/{METHOD}/${M}/g" \
      -e "s|{EXTRA_ARGS}|--method ${M} --datasets-dir ${DATASET_DIR}|g" \
      -e "s/{OUTPUT}/${DS}_${M}/g" \
      -e "s/{CPUS}/18/g" \
      -e "s/{MEM}/36G/g" \
      job_template.sbatch > "$JOB"
    sbatch "$JOB"
  done
done

# === DISTANCES ===
for DS in "${DATASETS[@]}"; do
  for D in "${DISTANCES[@]}"; do
    JOB="job_${DS}_ptdist_${D}.sbatch"
    sed \
      -e "s/{JOBNAME}/${DS}_ptdist_${D}/g" \
      -e "s/{DATASET}/${DS}/g" \
      -e "s/{MODE}/distance/g" \
      -e "s/{METHOD}/${D}/g" \
      -e "s|{EXTRA_ARGS}|--distance ${D} --datasets-dir ${DATASET_DIR}|g" \
      -e "s/{OUTPUT}/${DS}_ptdist_${D}/g" \
      -e "s/{CPUS}/1/g" \
      -e "s/{MEM}/4G/g" \
      job_template.sbatch > "$JOB"
    sbatch "$JOB"
  done
done

# === SPATIAL ENTROPY ===
for DS in "${DATASETS[@]}"; do
  JOB="job_${DS}_entropy.sbatch"
  sed \
    -e "s/{JOBNAME}/${DS}_entropy/g" \
    -e "s/{DATASET}/${DS}/g" \
    -e "s/{MODE}/entropy/g" \
    -e "s/{METHOD}/entropy/g" \
    -e "s|{EXTRA_ARGS}| --datasets-dir ${DATASET_DIR}|g" \
    -e "s/{OUTPUT}/${DS}_entropy/g" \
    -e "s/{CPUS}/1/g" \
    -e "s/{MEM}/4G/g" \
    job_template.sbatch > "$JOB"
  sbatch "$JOB"
done
