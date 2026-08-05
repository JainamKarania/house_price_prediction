# House Price Prediction

**Type:** Regression
**Status:** ✅ Done

## Problem
Predict median house value from structural, location, and demographic features.

## Dataset
California Housing dataset (`sklearn.datasets.fetch_california_housing`) — 8 features, ~20,640 rows, target `MedHouseVal` in units of $100,000.

## Models Tried
Linear Regression, Ridge, Lasso, ElasticNet

## Approach
1. EDA + correlation heatmap to check feature relationships and multicollinearity before modeling.
2. Train/test split (80/20) done **before** scaling; `StandardScaler` fit on train only, applied to both — avoids data leakage.
3. Baseline: plain Linear Regression on raw features, raw target.
4. Ridge and Lasso swept across a range of `alpha` values; inspected Lasso's coefficients to see which features got zeroed out.
5. Engineered 3 ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
6. Tried log-transforming the target (`MedHouseVal` is right-skewed) — see debugging findings below for why this was ultimately dropped.
7. Identified that ~4.3% of rows are top-coded/censored at the dataset's price ceiling ($500k) — excluded these from training only, kept them in the test set for an honest real-world evaluation.
8. Tuned alpha via `GridSearchCV` + 5-fold cross-validation on the final (no-log, engineered-feature) configuration — not just the log-transformed one it was originally selected under.
9. Residual plot as a sanity check beyond the aggregate metrics.

## Key Concepts I Can Explain
Bias-variance tradeoff, L1 vs L2 regularization, feature scaling (and why fit-on-train-only), MAE vs RMSE vs R² and when each is appropriate, multicollinearity and how Ridge/Lasso respond to it differently, k-fold cross-validation, GridSearchCV, censored/top-coded target data and how to handle it, why R² can worsen even when raw error improves (variance-explained denominator effect)

## Results

| Stage | RMSE | R² | Notes |
|---|---|---|---|
| Baseline: Linear Regression, raw features, raw target | 0.7456 | 0.5758 | Initial run |
| + log-transform, raw features (capped rows in training) | 0.9809 | 0.2657 | Log-transform hurt performance |
| + log-transform, engineered features, tuned Ridge, cap-excluded training | 0.8207 | 0.4860 | Engineered features helped, but log-transform still dragged it below baseline |
| **+ engineered features, tuned Ridge, cap-excluded training, NO log-transform** | **0.6815** | **0.6455** | **Final model — verified on local run** |

## Debugging Findings (the real value of this project)

1. **Silent variable-shadowing bug #1:** `y_train` vs `y_train_log`. Re-splitting the data created `y_train_log` but left a stale `y_train` from an earlier cell still in memory. `GridSearchCV`/`cross_val_score` were accidentally trained against the wrong-scale target, and `np.expm1()` was later applied to predictions that were never log-transformed in the first place — producing a nonsensical R² of roughly -1.8 million. No error was thrown; the bug was purely silent.
2. **Silent variable-shadowing bug #2:** engineered features (`X_engineered`) were computed but the re-split accidentally used the original `X`, not `X_engineered` — so three carefully engineered features were computed and then never used in training. Fixing this nearly doubled R² (0.2553 → 0.4860).
3. **Known dataset limitation:** `MedHouseVal` is top-coded at 5.00001 (~$500k). ~4.3% of rows sit exactly at this ceiling. Handled by excluding these rows from training (so the model isn't taught a fake ceiling) while keeping them in the test set (so evaluation reflects real-world deployment conditions).
4. **The "improvement" that wasn't:** the log-transformed model underperformed the plain baseline (R²: 0.4860 vs 0.5758). Ran a controlled, single-variable ablation (log-transform on/off, everything else held constant) and confirmed the log-transform itself — not the engineered features or regularization — was the cause. Final model drops it.
5. **R² vs RMSE can disagree:** excluding the highest-variance (capped) rows from an evaluation set can *lower* R² even while RMSE/MAE improve, because R² measures error relative to total variance in the data, and removing high-variance points shrinks the denominator faster than the numerator.

**Takeaway:** every technique used here (log-transform, regularization, feature engineering, cap handling) is a legitimate tool, but none of them help by default — each needs to be verified empirically against the specific dataset, not applied because it's "standard practice."

**Reproducibility note:** re-running this notebook locally (fresh environment, real `fetch_california_housing` download) produced MAE 0.4836, RMSE 0.6815, R² 0.6455 — matching the numbers above within floating-point/library-version noise. Confirms the fixes hold up outside the original debugging session.

## Next Steps
- Re-run against the real Kaggle "House Prices - Advanced Regression Techniques" dataset for a portfolio-ready, messier-data version
- Try tree-based models (Week 2) which don't require scaling or manual multicollinearity handling
