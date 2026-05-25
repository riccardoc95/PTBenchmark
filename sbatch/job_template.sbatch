#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task={CPUS}
#SBATCH --mem={MEM}


echo "Starting job for dataset={DATASET}, mode={MODE}, method={METHOD}"

micromamba run -n ptbenchmark ptbenchmark test \
    --dataset {DATASET} \
    --mode {MODE} \
    {EXTRA_ARGS} \
    --output results_partial/{OUTPUT}.h5

echo "Finished dataset={DATASET}, mode={MODE}, method={METHOD}"
