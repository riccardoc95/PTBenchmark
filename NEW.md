# Reviewer Experiments

This file lists only the new commands needed for the reviewer response, assuming
the older benchmark jobs have already been run.

## 1. Update The HPC Checkout

From the repository root on the HPC:

```bash
cd ~/PTBenchmark
git pull
git submodule update --init --recursive
conda run -n ptbenchmark pip install -e .
```

Check that the editable install points to the repository, not to a copied
`site-packages` version:

```bash
conda run -n ptbenchmark python -c "import ptbenchmark; print(ptbenchmark.__file__)"
```

The printed path should start with `~/PTBenchmark/ptbenchmark`.

## 2. Launch Only The Missing Reviewer Jobs

Run all new jobs:

```bash
bash sbatch/submit_missing_reviewer_jobs.sh
```

This submits:

- `nafnet`, `hirdiff`, `restormer` on all datasets.
- entropy stopping (`perstree_rec_sec`) on all datasets.
- sensitivity sweeps for `perstree` and `entropy` on all datasets.

The script submits jobs in batches of 10 by default.

Useful variants:

```bash
# Pilot sensitivity on 5 images per subset.
MAX_IMAGES=5 bash sbatch/submit_missing_reviewer_jobs.sh

# Skip entropy jobs if already completed.
RUN_ENTROPY=0 bash sbatch/submit_missing_reviewer_jobs.sh

# Run only NAFNet, HIR-Diff, and Restormer.
RUN_ENTROPY=0 RUN_SENSITIVITY=0 bash sbatch/submit_missing_reviewer_jobs.sh

# Run only sensitivity jobs.
RUN_NEW_SUPERVISED=0 RUN_ENTROPY=0 bash sbatch/submit_missing_reviewer_jobs.sh
```

## 3. Merge HDF5 Outputs

After all jobs finish:

```bash
ptbenchmark merge \
  --input-dir results_partial \
  --output results/results_all_merged.h5
```

If `ptbenchmark` is not on PATH inside the login shell:

```bash
micromamba run -n ptbenchmark ptbenchmark merge \
  --input-dir results_partial \
  --output results/results_all_merged.h5
```

## 4. Build Standard Result Summary

Generate the JSON summary and best-image plots:

```bash
micromamba run -n ptbenchmark python -m ptbenchmark.results.build_results
```

Generate LaTeX tables and plots:

```bash
micromamba run -n ptbenchmark python -m ptbenchmark.results.generate_tables_and_plots
```

These tables now include `NAFNet`, `HIR-Diff`, and `Restormer`.

## 5. Oracle Stopping vs Entropy Stopping

This directly addresses the stopping-criterion reviewer concern by comparing:

- `perstree`: oracle MSE stopping.
- `perstree_rec_sec`: automatic entropy stopping.

Run:

```bash
micromamba run -n ptbenchmark python -m ptbenchmark.results.analyze_oracle_vs_entropy \
  --input results/results_all_merged.h5 \
  --output-dir paper/analysis
```

Outputs:

```text
paper/analysis/oracle_vs_entropy_detail.csv
paper/analysis/oracle_vs_entropy_summary.csv
```

Use the summary CSV to report dataset/subset-level differences in:

- MSE
- PSNR
- SSIM
- runtime
- stopping iteration `niter`

## 6. Sensitivity Analysis Outputs

The missing-job script writes sensitivity results to:

```text
results_partial/<DATASET>_perstree_sensitivity.h5
results_partial/<DATASET>_perstree_sensitivity.csv
results_partial/<DATASET>_entropy_sensitivity.h5
results_partial/<DATASET>_entropy_sensitivity.csv
```

The default grid is:

```text
base_relax_scale: 1e-6, 1e-5, 1e-4
guided_radius:    1, 2, 3
epsilon_scale:    1e-4, 5e-4, 1e-3
stop_threshold:   1e-5, 1e-4, 1e-3  # entropy only
```

For `perstree`, this is `27` configurations per dataset.
For `entropy`, this is `81` configurations per dataset.

To run a custom narrower sweep:

```bash
BASE_RELAX_SCALES=1e-5 \
GUIDED_RADII=1,2 \
EPSILON_SCALES=1e-4,5e-4 \
STOP_THRESHOLDS=1e-5,1e-4 \
bash sbatch/submit_missing_reviewer_jobs.sh
```

## 7. Reviewer Mapping

Use the generated outputs as follows:

- Recent deep learning baselines: standard merged HDF5, tables including
  `nafnet`, `hirdiff`, `restormer`.
- Entropy stopping robustness: `oracle_vs_entropy_summary.csv`.
- First-significant-change criterion: report actual `stop_reason` and `niter`
  from entropy results.
- Filter usefulness: compare `perstree` vs `perstree_cut` in standard tables,
  plus RD/RF distance tables.
- Hyperparameter defaults: use sensitivity CSV files to show stability across
  the parameter grid.
- Runtime claims: standard runtime tables, now including the new supervised
  methods.
