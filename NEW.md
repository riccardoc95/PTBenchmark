# Reviewer Experiments

This file lists only the new commands needed for the reviewer response, assuming the older benchmark jobs have already been run.

## 1. Update The HPC Checkout

```bash
cd ~/PTBenchmark
git pull
git submodule update --init --recursive
ptbenchmark pip install -e .
```

## 2. Launch The Requests Jobs

Run all jobs:

```bash
bash sbatch/submit_all_jobs.sh
bash sbatch/submit_rev1.jobs.sh
```

This submits:

- `nafnet`, `hirdiff`, `restormer` on all datasets.
- entropy stopping (`perstree_rec_sec`) on all datasets.
- sensitivity sweeps for `perstree` and `entropy` on all datasets.

The script submits jobs in batches of 10 by default.

## 3. Merge HDF5 Outputs

After all jobs finish:

```bash
ptbenchmark merge \
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
ptbenchmark python -m ptbenchmark.results.analyze_oracle_vs_entropy \
  --input results/results_all_merged.h5 \
  --output-dir paper/analysis
```

Use the summary CSV to report dataset/subset-level differences in:

- MSE
- PSNR
- SSIM
- runtime
- stopping iteration `niter`
