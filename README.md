# Fashion-MNIST dimensionality-reduction benchmark

Reproducible comparison of PCA, RBF Kernel PCA, Gaussian SNE, t-SNE, and PCA→t-SNE. A k-NN classifier is trained in each learned representation. The same stratified samples and splits are reused for fairness.

## Run
```bash
python -m pip install -r requirements.txt
python run_experiment.py --config config.yaml
```
The loader first searches common Kaggle CSV/IDX locations and `FASHION_MNIST_DIR`; if absent it calls `kagglehub.dataset_download("zalando-research/fashionmnist")`.

## Outputs
- `outputs/tables/metrics.csv`, per-class metrics, timings, statistical tests
- `outputs/figures/`, embeddings, confusion matrices, ROC/FROC, timing and metric charts
- `report/Fashion_MNIST_DR_Report.docx`, automatically populated report

## Important design choices
- Non-parametric SNE and t-SNE do not naturally transform unseen observations. Therefore the benchmark embeds the combined stratified train/test sample, while the classifier is fit only on training rows. This is transductive and is stated in the report.
- Accuracy is not an intrinsic dimensionality-reduction metric. It is downstream k-NN accuracy in the embedding.
- “Loss” is method-specific: PCA reconstruction MSE, Kernel-PCA pre-image MSE when available, and KL divergence for SNE/t-SNE. These losses are not placed on one numerical scale.
- Full Kernel PCA and exact SNE are O(n²) in memory; defaults use a balanced 7,000-image benchmark.
