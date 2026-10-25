# Dimensionality reduction, manifold learning & feature selection on scRNA-seq (PBMC)

A learning tutorial that compares **dimensionality reduction** (PCA, Kernel PCA, MDS, Isomap, a hand-written SNE, t-SNE, UMAP) and **feature selection** (Variance, ANOVA F, Mutual Information, Fisher score, Lasso) on single-cell RNA-seq data of human blood cells (`scanpy.datasets.pbmc3k_processed`, 2,638 cells × 1,838 genes, 8 cell types).

Everything is evaluated with leakage-safe, repeated, stratified cross-validation, using three classifiers (k-NN, SVM, Random Forest), CPU and wall-clock timing, embedding-quality metrics and paired statistical tests. Every step is explained in plain language in the notebook.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset (~24 MB) is downloaded by scanpy on first use into `data/`.

## How to run

| What | Command | Time (8-core laptop) |
|---|---|---|
| Read & run the tutorial | `jupyter notebook tutorial_scrna_dr_fs.ipynb` | ~6 min (FAST) |
| Quick end-to-end check | `python run_pipeline.py --fast` | ~5 min |
| Full study | `python run_pipeline.py` | ~1–2 h (exact t-SNE/SNE/MDS at high dims dominate) |
| Only some parts | `python run_pipeline.py --parts baseline,B` | – |
| Notebook in full mode | `TUTORIAL_FAST=0 jupyter notebook …`, or set `FAST = False` in the first cell | – |

`tutorial_scrna_dr_fs.py` is the same notebook in jupytext "percent" format (readable in any editor, runnable cell by cell in VS Code/PyCharm). Re-sync after editing either file with `jupytext --sync tutorial_scrna_dr_fs.ipynb`.

## Configuration (`config.yaml`)

Everything is configurable:

* `dataset`: `pbmc3k_processed` | `pbmc68k_reduced` | `paul15`. Change one line to swap datasets. Add new loaders in `src/data.py::LOADERS`.
* `feature_source`: `lognorm_hvg` (log-normalised, unscaled HVGs; scaling happens inside each training fold) or `processed_X` (the pre-scaled matrix).
* `subsample_cells`, `clip_value`, `seed`, `cv.folds`, `cv.repeats`.
* `classifiers`: hyperparameters of k-NN, SVM and RF.
* `partA`: `methods`, `dims`, `pca_pre_dims`, `tsne_high_dim` (`exact` | `skip`), quality-metric settings.
* `hyper`: perplexities, iterations, `n_neighbors`, `min_dist`, Kernel PCA γ, …
* `partB`: `methods`, `k_values`, Lasso C, top-N genes, bootstrap count for stability.
* `partC`: which methods appear in the 2D grids.
* `markers`: canonical PBMC marker genes, used to annotate selected genes.
* `fast:` overrides applied by `--fast` (800 cells, 1 repeat, dims [2, 10], k [10, 50], fewer iterations, t-SNE > 3D skipped).

## Project layout

```
config.yaml                  all settings
run_pipeline.py              CLI: runs baseline, Part A, Part B, Part C
tutorial_scrna_dr_fs.ipynb   the tutorial (explanations + code + plots)
tutorial_scrna_dr_fs.py      same notebook, jupytext percent format
src/
  config.py      load_config (+ fast overrides), set_seed
  data.py        dataset registry/loaders, stratified subsample, scale+clip, summary
  reducers.py    make_reducer(): scale → [PCA-50] → method; inductive/transductive flags; timing
  sne.py         educational Gaussian SNE (vectorised gradient)
  selection.py   gene scorers (Variance, ANOVA, MI, Fisher, Lasso), TopKSelector, cache, Jaccard
  cv.py          classifiers, shared stratified splits, timed fit/predict
  metrics.py     classification metrics; trustworthiness, continuity, k-NN preservation, ARI, silhouette
  statistics.py  paired t-test / Wilcoxon vs baseline, one-way ANOVA across methods
  experiments.py run_data_summary / run_baseline / run_partA / run_partB / run_partC
  plots.py       every figure (fixed colour palette, 150 dpi)
```

## Key design choices

* **No leakage.** The scaler, PCA pre-step, reducer, gene selector and classifier are all fit on the training fold only (feature selection uses a sklearn `Pipeline`).
* **Transductive methods** (t-SNE, SNE, MDS) have no `transform()`, so train + test are embedded together. They are flagged `transductive=True` in the tables, marked "(T)" and dashed in the plots. Because train + test = all cells in every fold, the embedding is computed once per CV repeat (same result, 5× cheaper).
* **PCA-50 pre-step** before t-SNE, SNE, MDS, Isomap and UMAP: it denoises and cuts O(n²) costs (standard scanpy/Seurat practice).
* **Scale + clip at ±10** (as `sc.pp.scale(max_value=10)`), so rare genes cannot dominate distances.
* **t-SNE above 3 dims** uses `method='exact'` (full mode) or is skipped with a logged message (fast mode). Skips and failures are written to `partA_skipped_or_failed.csv`, and the pipeline keeps going.
* **Timing.** `time.process_time()` (CPU, all threads) and `time.perf_counter()` (wall) are recorded separately for reducer fit, reducer transform, selection, classifier training and prediction.
* **Same folds everywhere**, so method-vs-baseline tests are paired fold by fold.

## Output files

Each mode writes to its own folder, so a fast run never overwrites full results: `outputs/full/` (`python run_pipeline.py`, or the notebook with `FAST = False`) and `outputs/fast/` (`--fast`, or the notebook's default). Inside each, tables are in `tables/` and figures (150 dpi PNG) in `figures/`; the paths below are relative to that folder.

### Data & baseline
| File | Contents |
|---|---|
| `tables/data_summary.csv` | cells per cell type and fraction |
| `figures/data_class_counts.png` | bar chart of cells per type (class imbalance) |
| `figures/prep_gene_variance.png` | *(notebook)* gene variances before vs after z-scoring |
| `tables/baseline_results_folds.csv` | every fold × classifier on all genes: accuracy, macro P/R/F1, κ, ROC AUC, CPU & wall times |
| `tables/baseline_summary.csv` | mean and sd of the above per classifier |
| `figures/baseline_accuracy_f1.png` | accuracy & macro-F1 per classifier, ± sd error bars |
| `figures/baseline_cpu_time.png` | training & prediction CPU time per classifier |

### Part A: dimensionality reduction
| File | Contents |
|---|---|
| `tables/partA_results_folds.csv` | every method × dim × classifier × fold: metrics, `transductive` flag, reducer fit/transform CPU & wall, classifier CPU & wall, totals |
| `tables/partA_summary.csv` | means per method × dim × classifier, plus `delta_accuracy`, `delta_macro_f1` and `speedup_cpu` vs baseline |
| `tables/partA_embedding_quality.csv` | trustworthiness, continuity, k-NN preservation, k-means ARI, silhouette per method × dim × repeat |
| `tables/partA_stats_vs_baseline.csv` | paired t-test and Wilcoxon (accuracy, macro-F1) vs baseline per method × dim × classifier |
| `tables/partA_anova.csv` | one-way ANOVA across methods per classifier × dim |
| `tables/partA_skipped_or_failed.csv` | method/dim combinations skipped or failed, with the reason |
| `figures/partA_accuracy_vs_dim.png`, `partA_f1_vs_dim.png` | metric vs target dims, one line per method, baseline dotted |
| `figures/partA_delta_accuracy_heatmap.png` | method × dims grid of accuracy change vs baseline |
| `figures/partA_cpu_time.png` | reducer CPU vs dims, and total-pipeline speed-up vs baseline |
| `figures/partA_quality_vs_dim.png` | the five embedding-quality metrics vs dims |
| `figures/partA_pca_scree.png` | *(notebook)* PCA explained variance + 2D PCA |
| `figures/partA_kpca_gamma.png` | *(notebook)* Kernel PCA for three γ values |
| `figures/partA_mds_shepard.png` | *(notebook)* MDS Shepard diagram + 2D MDS |
| `figures/partA_isomap_neighbors.png` | *(notebook)* Isomap for three `n_neighbors` |
| `figures/partA_sne_loss.png` | *(notebook)* SNE KL-divergence curve + 2D SNE |
| `figures/partA_tsne_perplexity.png` | *(notebook)* t-SNE for three perplexities |
| `figures/partA_umap_params.png` | *(notebook)* UMAP `n_neighbors` × `min_dist` |

### Part B: feature selection
| File | Contents |
|---|---|
| `tables/partB_results_folds.csv` | every selector × k × classifier × fold: metrics, selection/train/predict CPU & wall |
| `tables/partB_summary.csv` | means per selector × k × classifier, with deltas and speed-up vs baseline |
| `tables/partB_selector_cpu.csv` | CPU/wall time to score all genes, per selector × fold |
| `tables/partB_stats_vs_baseline.csv`, `partB_anova.csv` | paired tests vs baseline; ANOVA across selectors |
| `tables/partB_top20_genes.csv` | top-20 genes per selector (rank, score, `known_marker`) |
| `tables/partB_marker_hits.csv` | how many top-20 genes per selector are canonical markers, and which |
| `tables/partB_method_jaccard.csv` | Jaccard overlap of top-`overlap_k` gene sets between selectors |
| `tables/partB_stability.csv` | mean ± sd bootstrap Jaccard of each selector's top-k set |
| `figures/partB_accuracy_vs_k.png`, `partB_f1_vs_k.png` | metric vs number of genes, one line per selector |
| `figures/partB_heatmap_accuracy.png` | selector × classifier accuracy, one panel per k |
| `figures/partB_cpu_time.png` | scoring cost per selector; total CPU vs k |
| `figures/partB_top20_genes.png` | top-20 genes per selector, markers starred |
| `figures/partB_method_jaccard.png` | agreement heatmap between selectors |
| `figures/partB_stability.png` | bootstrap stability per selector |
| `figures/partB_method_*.png` | *(notebook)* sorted scores + top-gene expression per type, per selector |

### Part C: 2D visualisation
| File | Contents |
|---|---|
| `tables/partC_2d_metrics.csv` | per 2D method: CPU/wall time, trustworthiness, continuity, k-NN preservation, ARI, silhouette, 5-fold k-NN accuracy on the 2D coordinates |
| `figures/partC_2d_grid_dr.png` | 2D maps from all 7 reducers, same palette, shared legend, metrics under each panel |
| `figures/partC_2d_grid_fs.png` | 2D maps from the top-2 genes of each selector |
| `figures/partC_2d_summary.png` | bar chart comparing the 2D methods on trustworthiness, continuity, ARI and CPU time |
