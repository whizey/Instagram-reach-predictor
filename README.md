# Social Media Engagement Forecasting

A machine learning and analytics platform that estimates an Instagram post's
total reach (impressions) from its engagement signals, and surfaces which
behaviors drive visibility. The trained model is served in production through a
Django backend.

> **What this does:** given a post's engagement (likes, saves, comments, shares,
> profile visits, follows), it estimates the post's total impressions and breaks
> down the engagement that matters most. It is a *reach-estimation and analysis*
> tool, not a pre-publication forecaster — the inputs are signals a post
> accumulates after going live.

**Best model:** Ridge Regression — test R² **0.91**, 5-fold CV R² **0.72–0.81**
(fold-dependent on a 119-row set), MAE ~1,261, and **71% lower error than a
mean baseline**.

---

## Project Overview

Creators often judge a post by likes alone, but likes don't explain why some
posts reach far more people than others. This project asks a focused question:

> Can a post's total reach be estimated from its engagement signals, and which
> signals carry the most weight?

The system includes:

- A **machine learning pipeline** comparing five regression models for impression estimation
- An **analytics engine** for viral scoring, engagement rate, and follow conversion
- A **Django backend** that serves the trained model via API and stores per-session history

---

## Dataset

- Source: public Instagram post-level engagement dataset (119 posts, single account)
- One row per post, with raw engagement counts plus caption and hashtag text
- **Target:** `Impressions` (total reach per post)

Because the dataset is small and single-account, results are reported with
cross-validation, and generalization to other accounts would require retraining.

---

## A note on data leakage (important)

The raw dataset includes columns that break impressions down by source
(`From Home`, `From Hashtags`, `From Explore`). These were **excluded** — they
are components of the target and would leak it.

An earlier version also engineered an `engagement_rate = total_engagement /
Impressions` feature. Because it divides by the target, it inflated R² to ~0.95.
**It was removed and the model retrained on leak-free features**, which is why the
honest R² reported here (0.91 test / 0.72 CV) is lower than an early draft — and
far more trustworthy.

---

## Features used (leak-free)

Raw signals plus a few derived ratios that capture behavior, none of which touch
the target:

| Feature | Source |
|---|---|
| Likes, Saves, Comments, Shares, Profile Visits, Follows | raw engagement |
| Save-to-Like ratio | Saves / (Likes + 1) |
| Share-to-Like ratio | Shares / (Likes + 1) |
| Follow rate | Follows / (Profile Visits + 1) |
| Total engagement | Likes + Saves + Comments + Shares |
| Hashtag count | parsed from caption |
| Caption length | character count |

---

## Model Training

Five regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Gradient Boosting
- Random Forest

Configuration: StandardScaler normalization, 80/20 train–test split, 5-fold
cross-validation for generalization.

---

## Results — Impressions

| Model | Test R² | CV R² | MAE | RMSE |
|---|---|---|---|---|
| **Ridge Regression** | **0.914** | **0.720** | 1,261 | 1,828 |
| Lasso Regression | 0.904 | 0.688 | 1,344 | 1,928 |
| Linear Regression | 0.903 | 0.686 | 1,352 | 1,945 |
| Gradient Boosting | 0.922 | 0.745 | 896 | 1,738 |
| Random Forest | 0.872 | 0.684 | 1,122 | 2,229 |

**Why Ridge was chosen.** Gradient Boosting edged it on CV R² (0.745 vs 0.720),
but on a 119-row dataset that margin is within noise, and Ridge is more stable
and interpretable. The interpretability/stability trade-off favored Ridge for a
model that has to be explained and trusted, so Ridge was shipped.

---

## Ridge — detailed metrics

**Target context** (so the errors mean something): impressions average **5,704**,
median **4,289**, ranging **1,941 – 36,919** (std 4,844).

| Metric | Value | Reading |
|---|---|---|
| Test R² | 0.914 | explains ~91% of variance on held-out posts |
| Train R² | 0.906 | **gap ≈ 0** → no meaningful overfitting |
| Test MAE | 1,261 | average miss in absolute impressions |
| Test RMSE | 1,828 | error with large misses penalized |
| Test MAPE | 20.7% | average miss as a % of actual reach |
| vs mean baseline | **−71% RMSE** | beats predict-the-average by 71% |

**Cross-validation (5-fold):** per-fold R² = [0.91, 0.75, 0.81, 0.79, 0.81],
mean **0.81**, std **0.05**. (A contiguous, non-shuffled split lands nearer 0.72;
the spread reflects the small dataset, so the honest range is ~0.72–0.81.)

**Feature influence** (standardized Ridge coefficients): **Saves** and **Profile
Visits** are the dominant positive drivers of predicted reach, followed by total
engagement — consistent with the headline insight that saves signal reach more
than likes. Note: several engagement features are correlated, so individual
coefficient *signs* on the weaker features are not stable enough to
over-interpret; the reliable takeaway is the dominance of saves and profile
visits.

---

## Key Insight

Saves and follower growth are stronger predictors of reach than likes or
comments — the algorithm appears to reward signals of genuine intent:

- Saves → content worth revisiting
- Shares → content worth spreading
- Follows → content worth subscribing to

**Takeaway for creators:** optimize for saves and follow conversion, not just likes.

---

## Serving

The trained Ridge pipeline (scaler + model) is serialized to
`analytics/models/ridge_pipeline.pkl` and loaded by the Django app at startup.
Each `/api/analyze/` request builds the feature row and returns the model's
prediction. A rule-based fallback runs only if the model artifact is missing, so
the API stays available either way.
