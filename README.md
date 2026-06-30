# 📊 Instra — Instagram Reach Predictor

Predict how many people an Instagram post will reach, and understand *why* —
powered by a trained machine learning model, served through a Django web app.

> **In one line:** you give it a post's engagement (likes, saves, comments,
> shares…), and it estimates the post's total **impressions** (reach) plus a
> breakdown of what's driving that reach.

🔗 **Live demo:** https://instagram-ml-django.onrender.com

---

## Table of Contents
1. [What this project does](#what-this-project-does)
2. [How it works (plain English)](#how-it-works-plain-english)
3. [Tech stack](#tech-stack)
4. [Quick start — run it locally](#quick-start--run-it-locally)
5. [How to retrain the model](#how-to-retrain-the-model)
7. [API reference](#api-reference)
8. [The model & metrics](#the-model--metrics)
9. [A note on data leakage](#a-note-on-data-leakage)
10. [Key insight](#key-insight)
11. [Limitations & honest caveats](#limitations--honest-caveats)

---

## What this project does

Content creators usually judge a post by its likes. But likes don't explain why
one post reaches 3,000 people and another reaches 30,000. This project answers a
focused question:

> **Can we estimate a post's total reach from its engagement signals — and which
> signals matter most?**

It has three parts working together:

- 🤖 **A machine learning model** that estimates impressions from engagement.
- 📈 **An analytics engine** that scores virality, engagement rate, and follower conversion.
- 🌐 **A Django web app** that wraps the model in a clean UI and a JSON API.

---

## How it works 

Think of it as a four-stage assembly line. A post's numbers go in one end, an
estimate comes out the other.

**1. Raw data in.** Each post is a row of numbers: likes, saves, comments,
shares, profile visits, follows, plus the caption and hashtags.

**2. Feature engineering.** Raw counts alone don't tell the full story, so we
build extra signals from them — for example *saves-per-like* (are people
bookmarking, or just tapping ❤️?) and *follow-rate* (do profile visitors
actually follow?). These ratios capture *behavior*, not just volume.

**3. The model makes an estimate.** A trained **Ridge Regression** model has
already "studied" 119 real posts and learned the relationship between engagement
and reach. It takes the features and outputs an impression estimate.

**4. Insights on top.** The app adds a viral score, identifies the strongest
levers, and suggests improvements.

> **Important framing:** the inputs (likes, saves, …) are things a post collects
> *after* it goes live. So this is a **reach-estimation and analysis** tool, not
> a crystal ball that predicts a post before you publish it.

### How the model "learned" (the ML part, simply)

Training a model is like teaching by example:

1. We showed it 119 posts where we knew both the engagement **and** the real
   impressions (the answer).
2. It adjusted its internal numbers until its guesses matched the real answers
   as closely as possible. This is **`.fit()`** — the actual "training."
3. We held back 20% of posts it never saw during training, then tested on those.
   This checks it actually *learned the pattern* instead of memorizing.
4. We compared **5 different model types** and kept the best one.

---

## Tech stack

| Layer | Tools |
|---|---|
| ML | scikit-learn (Ridge, Lasso, Linear, Gradient Boosting, Random Forest), pandas, NumPy |
| Backend | Django, Django REST views |
| Model serving | joblib (serialized scaler + model pipeline) |
| AI advisor | Groq (Llama 3.3 70B) with a Gemini 2.0 Flash fallback |
| Deployment | Render, Gunicorn |
| Storage | SQLite (per-session post history) |

---

## Quick start — run it locally

You need **Python 3.10+** installed. Then:

### 1. Clone the repo
```bash
git clone https://github.com/whizey/Instagram-reach-predictor.git
cd Instagram-reach-predictor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# activate it:
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
A virtual environment keeps this project's packages separate from the rest of
your computer, so nothing clashes.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
This installs Django, scikit-learn, and everything else the app needs.

### 4. Set up the database
```bash
python manage.py migrate
```
This creates the SQLite tables that store post history.

### 5. (Optional) Add an AI advisor key
The app works fully without this — it falls back to smart rule-based advice. To
enable LLM-powered suggestions:
```bash
export GROQ_API_KEY=your-key-here      # Mac/Linux
set GROQ_API_KEY=your-key-here         # Windows
```

### 6. Start the server
```bash
python manage.py runserver
```
Open **http://127.0.0.1:8000** in your browser. Enter a post's numbers, hit
analyze, and you'll get a predicted reach + insights. 🎉

---

## How to retrain the model

You only need this if you change the data or features. The app ships with a
pre-trained model (`analytics/models/ridge_pipeline.pkl`), so **normal use needs
no retraining.**

To regenerate the model from scratch:

```bash
python train_model.py
```

This script:
1. Loads `Instagram_Data.csv`
2. Rebuilds the leak-free features
3. Trains and compares all 5 models
4. Saves the best one (Ridge) to `analytics/,M,ridge_pipeline.pkl`

The saved `.pkl` file *is* the trained model — it contains the learned weights
and the scaler together, so the web app just loads it and runs predictions. No
training happens at request time.

> If you don't have a `train_model.py` yet, the training logic also lives in
> `Instagram Reach Viral Prediction.ipynb` (open it in Jupyter and run all cells).

---

## API reference

| Method | Endpoint | What it does |
|---|---|---|
| POST | `/api/analyze/` | Analyze a post → impressions + viral score + advice |
| POST | `/api/history/` | Get this session's past posts |
| POST | `/api/clear/` | Clear session history |
| POST | `/api/agent/` | Chat with the AI strategy advisor |

### Example: analyze a post
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_key": "my-session",
    "likes": 162, "saves": 98, "comments": 9,
    "shares": 5, "follows": 2, "profile_visits": 35,
    "caption_length": 150, "hashtags": 20, "reposts": 0
  }'
```
Returns predicted impressions, viral score, engagement rate, and growth tips as
JSON.,,K

---

## The model & metrics

Five regression models were trained and compared. **Ridge Regression won.**

### Impressions — all models

| Model | Test R² | CV R² | MAE | RMSE |
|---|---|---|---|---|
| **Ridge Regression** ✅ | **0.914** | **0.72–0.81** | 1,261 | 1,828 |
| Lasso Regression | 0.904 | 0.688 | 1,344 | 1,928 |
| Linear Regression | 0.903 | 0.686 | 1,352 | 1,945 |
| Gradient Boosting | 0.922 | 0.745 | 896 | 1,738 |
| Random Forest | 0.872 | 0.684 | 1,122 | 2,229 |

### Ridge — detailed report

To make the numbers meaningful: impressions in this dataset average **5,704**,
median **4,289**, ranging **1,941 – 36,919**.

| Metric | Value | What it means |
|---|---|---|
| Test R² | 0.914 | explains ~91% of the variation in reach |
| Train R² | 0.906 | **gap ≈ 0 → no overfitting** |
| MAE | 1,261 | predictions are off by ~1,261 impressions on average |
| RMSE | 1,828 | like MAE, but punishes big misses harder |
| MAPE | 20.7% | average miss as a % of the real value |
| vs. baseline | **−71% RMSE** | beats just guessing the average by 71% |

**What these mean in plain words:**
- **R²** = how much of the "why posts differ" the model captures. 0.91 is strong.
- **The near-zero train/test gap** is the headline — it means the model didn't
  just memorize the 119 posts; it generalizes to new ones.
- **The 71% beat over baseline** proves the model adds real value, not just a
  pretty R² on an easy target.

**Why Ridge over Gradient Boosting?** GB scored slightly higher on CV (0.745 vs
0.720), but on only 119 rows that gap is within noise — and Ridge is more stable
and far easier to explain. For a model that has to be trusted, interpretability
won.

---

## A note on data leakage

This is the most important engineering lesson in the project, so it's worth
stating plainly.

The raw data contained columns that broke impressions down by source
(`From Home`, `From Hashtags`, `From Explore`). These literally add up to the
target — using them would be like predicting someone's total bill from the
individual line items. **They were excluded.**

An earlier version also engineered `engagement_rate = total_engagement /
Impressions` — which divides by the target. That single leak inflated R² to
**~0.95**. It was caught, removed, and the model retrained, giving the honest
**0.91** reported here.

A lower, leak-free score you can trust beats an impressive number you can't. 👍

---

## Key insight

> **Saves and follower growth predict reach better than likes or comments.**

The model's strongest signals were **Saves** and **Profile Visits** — suggesting
the algorithm rewards genuine intent over passive reactions:

- 🔖 **Saves** → "I want to come back to this"
- 🔁 **Shares** → "Others should see this"
- ➕ **Follows** → "I want more from this creator"

**Takeaway for creators:** design for saves and follows, not just likes.

*(Caveat: several engagement signals are correlated, so we read the dominant
drivers — saves, profile visits — rather than over-interpreting every individual
coefficient.)*

---
