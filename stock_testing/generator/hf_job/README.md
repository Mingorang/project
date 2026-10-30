# Weighted-vs-uniform training job

Fits `HistGradientBoostingRegressor` twice on `$SIM_ARCHIVE_REPO` — once with the
accuracy-derived `sample_weight`, once uniform — and reports the out-of-sample R² delta with a
bootstrap 95% CI. The dataset was produced by `export_hf_dataset.py --scored-only` from the
`stock_testing/generator` tree; the exact commit and `config.toml` hash are recorded in
`dataset/dataset_info.json` and `dataset/data_card.md` in the same repo. Re-run locally with
`python train_weighted.py --repo <repo> --out model_out`. Artifacts land in
`runs/<job_id>/` in the dataset repo; the job's own `/results` is ephemeral.
