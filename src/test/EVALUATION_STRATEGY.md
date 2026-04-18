# Evaluation Strategy & Significance

This document outlines the testing and evaluation strategy implemented in `src/test/evaluate_models.py` and explains its significance for estimating the true performance of the models in the final platform evaluation.

## 1. The Challenge of "Test" Data
In typical competition environments or blind-test deployments, the official `test` split provided to participants contains no ground-truth labels (e.g., `forest_mask`, `consensus`, `weight`). This prevents local scripts from calculating metrics like Intersection over Union (IoU) or F1-Score on the actual test data.

To circumvent this and establish a reliable local leaderboard, we must evaluate our models on a **hold-out validation set** derived from the `train` split, where ground-truth labels are available.

## 2. Our Local Evaluation Pipeline
The `evaluate_models.py` script automates this process across all trained models (checkpoints):

1.  **Iterating the Training Cache:** Instead of failing on the label-less `test` cache, the script automatically targets the `train` cache.
2.  **Hold-out Set Enforcement:** Each checkpoint (`best.pt`) contains metadata defining the specific geographical regions (e.g., `['48P']`) that were held out as validation data during its training. The evaluation script strictly respects this boundary. It tests Model A **only** on the tiles belonging to Model A's validation regions.
3.  **Dynamic Feature Filtering:** Because different models might use different sets of input features (e.g., a simple baseline using 264 features vs. a temporal model using 682 features), the script dynamically builds the superset feature cube and dynamically slices out the specific `feature_names` expected by each model's checkpoint.

## 3. Metrics Computed
The pipeline evaluates pixel-level predictions against the local ground truth using two primary metrics:

*   **Weighted F1-Score / w-IoU:** This takes into account the consensus weights (`weights > 0`). Pixels where multiple annotators agree have a higher impact on the score.
*   **Unweighted IoU (u-IoU):** This is a raw pixel-overlap metric (`True Positives / (True Positives + False Positives + False Negatives)`). **This is the most significant metric**, as it most closely mirrors the unweighted, binary mask evaluation typically performed by external grading platforms.

## 4. How Representative is this Score?
The generated local leaderboard provides a **highly representative estimate** of the final submission score, with a few caveats:

### Why it is representative:
*   **Strict Data Separation:** Because we evaluate strictly on the hold-out validation regions saved during the training phase, the models have never seen the evaluated pixels during their optimization. This prevents data leakage and overfitting bias.
*   **Identical Post-Processing:** The evaluation script applies the exact same morphological closing, opening, and area-filtering logic (`PostprocessConfig`) that the final `infer.py` submission script uses.
*   **Unweighted IoU Focus:** By sorting the local leaderboard by `unweighted_iou`, we align our optimization goal with typical platform grading scripts that treat all submitted polygons equally.

### Potential Variances (Caveats):
1.  **Regional Shift:** If the geographical distribution of the true blind test set differs significantly from your chosen validation regions (e.g., the validation set is mostly in Southeast Asia, but the hidden test set contains dense Amazonian rainforest), the local score might slightly overestimate or underestimate true performance due to domain shift.
2.  **Ground Truth Confidence:** Our local validation labels use `weight` maps. The external platform's "hidden" ground truth might rely on slightly different consensus thresholds or newer annotation passes, leading to minor deviations in the raw IoU.
3.  **Polygon Area Filtering:** Small geometric differences between raster-based IoU (calculated here) and vector-based IoU (calculated on GeoJSON polygons by the platform) can cause fractional deviations, especially near the `min_area_ha` threshold.

## 5. Conclusion
The current `evaluate_models.py` pipeline is the most robust way to locally compare architectures, hyperparameters, and feature sets. The model that achieves the highest `u-IoU` across its validation regions is statistically the most likely to perform best on the final, hidden test set.
