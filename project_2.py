"""
=============================================================
  PROJECT 2 — House Price Prediction (End-to-End ML Project)
=============================================================
  Dataset  : Kaggle - House Prices: Advanced Regression Techniques
  Target   : SalePrice (continuous)
  Models   : Linear Regression | Random Forest | Gradient Boosting
  Metrics  : RMSE | MAE | R²
=============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os, random
random.seed(42)
np.random.seed(42)

SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET (mirrors Kaggle structure)
# ─────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — END-TO-END ML PROJECT")
print("=" * 60)
print("\n[1] Generating synthetic dataset (Kaggle-compatible structure)...")

N = 1460  # same as Kaggle training set

neighborhoods   = ["NAmes","CollgCr","OldTown","Edwards","Somerst","Gilbert","NridgHt","Sawyer","NWAmes","SawyerW"]
house_styles    = ["1Story","2Story","1.5Fin","SLvl","SFoyer"]
bldg_types      = ["1Fam","TwnhsE","Twnhs","Duplex","2fmCon"]
qualities       = ["Ex","Gd","TA","Fa","Po"]
conditions      = ["Norm","Feedr","PosN","Artery","RRAn"]
garage_types    = ["Attchd","Detchd","BuiltIn","CarPort","None"]
sale_conditions = ["Normal","Partial","Abnorml","Family","Alloca"]

df = pd.read_csv("train.csv")
print(f"Dataset loaded: {df.shape}")
print(df.head())

# Generate realistic SalePrice based on key features
quality_map = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1}
nbr_premium = {n: v for n, v in zip(neighborhoods,
    [0.0, 0.1, -0.12, -0.08, 0.18, 0.05, 0.30, -0.05, 0.02, 0.07])}

base_price = (
    50000
    + df["OverallQual"]  * 12000
    + df["GrLivArea"]    * 55
    + df["TotalBsmtSF"]  * 20
    + df["GarageCars"].fillna(0) * 8000
    + df["Fireplaces"]   * 6000
    + (2010 - df["YearBuilt"]).clip(0) * (-250)
    + df["FullBath"]     * 4000
    + df["BedroomAbvGr"] * 3000
    + df["WoodDeckSF"]   * 20
    + df["OpenPorchSF"]  * 15
    + df["Neighborhood"].map(nbr_premium).fillna(0) * 50000
    + np.random.normal(0, 18000, N)
).clip(34900, 755000)

df["SalePrice"] = base_price.astype(int)

print(f"   Dataset shape     : {df.shape}")
print(f"   Numeric features  : {df.select_dtypes(include=np.number).shape[1]}")
print(f"   Categorical feats : {df.select_dtypes(include='object').shape[1]}")
print(f"   Missing values    : {df.isnull().sum().sum()}")
print(f"\n   SalePrice statistics:")
print(df["SalePrice"].describe().to_string())

# ─────────────────────────────────────────────
# FIGURE 1 — EDA Overview (2×2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 1 — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)
palette = "#2563EB"

# 1a: SalePrice distribution
ax = axes[0, 0]
ax.hist(df["SalePrice"], bins=50, color=palette, edgecolor="white", alpha=0.85)
ax.set_title("Distribution of SalePrice", fontsize=13, fontweight="bold")
ax.set_xlabel("Sale Price ($)")
ax.set_ylabel("Count")
ax.axvline(df["SalePrice"].median(), color="#DC2626", lw=2, ls="--", label=f"Median: ${df['SalePrice'].median():,.0f}")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

# 1b: Log SalePrice distribution
ax = axes[0, 1]
log_price = np.log1p(df["SalePrice"])
ax.hist(log_price, bins=50, color="#7C3AED", edgecolor="white", alpha=0.85)
ax.set_title("Log-Transformed SalePrice", fontsize=13, fontweight="bold")
ax.set_xlabel("log(SalePrice + 1)")
ax.set_ylabel("Count")

# 1c: Overall Quality vs SalePrice
ax = axes[1, 0]
qual_means = df.groupby("OverallQual")["SalePrice"].median()
bars = ax.bar(qual_means.index, qual_means.values, color=plt.cm.RdYlGn(
    np.linspace(0.1, 0.9, len(qual_means))), edgecolor="white")
ax.set_title("Median SalePrice by Overall Quality", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Quality (1–10)")
ax.set_ylabel("Median Sale Price ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

# 1d: GrLivArea vs SalePrice scatter
ax = axes[1, 1]
sc = ax.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.25, c=df["OverallQual"],
                cmap="RdYlGn", s=12)
plt.colorbar(sc, ax=ax, label="Overall Quality")
ax.set_title("Living Area vs SalePrice", fontsize=13, fontweight="bold")
ax.set_xlabel("Above Ground Living Area (sq ft)")
ax.set_ylabel("Sale Price ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig1_eda.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[✓] Figure 1 saved — EDA Overview")

# ─────────────────────────────────────────────
# FIGURE 2 — Missing Values & Correlations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 2 — Missing Values & Feature Correlations", fontsize=15, fontweight="bold")

# 2a: Missing values bar chart
ax = axes[0]
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False).head(15)
colors_m = ["#DC2626" if v/N > 0.15 else "#F59E0B" if v/N > 0.05 else "#3B82F6"
            for v in missing.values]
bars = ax.barh(missing.index, missing.values / N * 100, color=colors_m, edgecolor="white")
ax.set_xlabel("Missing (%)")
ax.set_title("Top 15 Features with Missing Values", fontsize=12, fontweight="bold")
for bar, val in zip(bars, missing.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val}", va="center", fontsize=9)
ax.axvline(15, color="#DC2626", ls="--", alpha=0.5, label="15% threshold")
ax.legend(fontsize=9)

# 2b: Correlation heatmap of top numeric features
ax = axes[1]
top_num_cols = ["SalePrice","OverallQual","GrLivArea","TotalBsmtSF","1stFlrSF",
                "GarageArea","YearBuilt","FullBath","TotRmsAbvGrd","Fireplaces",
                "LotArea","BedroomAbvGr"]
corr = df[top_num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, annot_kws={"size": 8},
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap (Top Numeric Features)", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig2_missing_corr.png", dpi=130, bbox_inches="tight")
plt.close()
print("[✓] Figure 2 saved — Missing Values & Correlations")

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2] Data Preprocessing...")

df_clean = df.copy()

# 2a — Handle missing values
# Numeric: median imputation
num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != "SalePrice" and c != "Id"]

for col in num_cols:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# Categorical: fill with "None" (no feature) or mode
cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
none_fill_cols = ["BsmtQual","BsmtCond","FireplaceQu","GarageType","GarageFinish"]
for col in cat_cols:
    if col in none_fill_cols:
        df_clean[col] = df_clean[col].fillna("None")
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Also handle any remaining NaN in all columns (catch-all)
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].fillna("Unknown")
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"   Missing values after imputation: {df_clean.isnull().sum().sum()}")

# 2b — Remove duplicates
dup_before = len(df_clean)
df_clean.drop_duplicates(subset=[c for c in df_clean.columns if c != "Id"], inplace=True)
print(f"   Duplicates removed: {dup_before - len(df_clean)}")

# 2c — Outlier removal (GrLivArea > 4000 and price < 300K — known Kaggle issue)
mask_outlier = ~((df_clean["GrLivArea"] > 4000) & (df_clean["SalePrice"] < 200000))
removed = (~mask_outlier).sum()
df_clean = df_clean[mask_outlier].reset_index(drop=True)
print(f"   Outliers removed  : {removed}")
print(f"   Remaining rows    : {len(df_clean)}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Feature Engineering...")

df_clean["HouseAge"]      = df_clean["YrSold"] - df_clean["YearBuilt"]
df_clean["RemodAge"]      = df_clean["YrSold"] - df_clean["YearRemodAdd"]
df_clean["WasRemodeled"]  = (df_clean["YearBuilt"] != df_clean["YearRemodAdd"]).astype(int)
df_clean["TotalSF"]       = df_clean["TotalBsmtSF"] + df_clean["1stFlrSF"] + df_clean["2ndFlrSF"]
df_clean["TotalBaths"]    = (df_clean["FullBath"]
                            + 0.5 * df_clean["HalfBath"]
                            + df_clean["BsmtFullBath"]
                            + 0.5 * df_clean["BsmtHalfBath"])
df_clean["TotalPorchSF"]  = df_clean["OpenPorchSF"] + df_clean["EnclosedPorch"] + df_clean["WoodDeckSF"]
df_clean["HasGarage"]     = (df_clean["GarageArea"] > 0).astype(int)
df_clean["HasBsmt"]       = (df_clean["TotalBsmtSF"] > 0).astype(int)
df_clean["HasFireplace"]  = (df_clean["Fireplaces"] > 0).astype(int)
df_clean["QualTimesArea"] = df_clean["OverallQual"] * df_clean["GrLivArea"]
df_clean["IsNew"]         = (df_clean["YearBuilt"] >= 2000).astype(int)

eng_feats = ["HouseAge","RemodAge","WasRemodeled","TotalSF","TotalBaths",
             "TotalPorchSF","HasGarage","HasBsmt","HasFireplace","QualTimesArea","IsNew"]
print(f"   Engineered features added: {eng_feats}")

# ─────────────────────────────────────────────
# 4. ENCODING & SCALING
# ─────────────────────────────────────────────
print("\n[4] Encoding Categorical Features & Scaling...")

df_model = df_clean.drop(columns=["Id"])
cat_cols_model = df_model.select_dtypes(include="object").columns.tolist()
df_model = pd.get_dummies(df_model, columns=cat_cols_model, drop_first=True)

# Rename bool columns
bool_cols = df_model.select_dtypes(include="bool").columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

print(f"   Shape after one-hot encoding: {df_model.shape}")

# ─────────────────────────────────────────────
# 5. FEATURE SELECTION & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
print("\n[5] Feature Selection & Train/Test Split...")

X = df_model.drop(columns=["SalePrice"])
y = np.log1p(df_model["SalePrice"])   # log-transform target

# Remove near-zero variance columns
var = X.var()
X = X.loc[:, var > 0.01]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale numeric features (for Linear Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Training set : {X_train.shape[0]} rows × {X_train.shape[1]} features")
print(f"   Test set     : {X_test.shape[0]} rows × {X_test.shape[1]} features")

# ─────────────────────────────────────────────
# FIGURE 3 — Feature Engineering Insights
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure 3 — Feature Engineering Insights", fontsize=15, fontweight="bold")

ax = axes[0]
ax.scatter(df_clean["TotalSF"], df_clean["SalePrice"], alpha=0.2, color="#2563EB", s=10)
ax.set_xlabel("Total Square Footage"); ax.set_ylabel("Sale Price ($)")
ax.set_title("TotalSF vs SalePrice", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

ax = axes[1]
age_bins = pd.cut(df_clean["HouseAge"], bins=[0,10,20,30,50,75,100,150], right=False)
df_clean.groupby(age_bins)["SalePrice"].median().plot(kind="bar", ax=ax, color="#7C3AED", edgecolor="white")
ax.set_xlabel("House Age (years)"); ax.set_ylabel("Median Sale Price ($)")
ax.set_title("HouseAge vs SalePrice", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

ax = axes[2]
bath_means = df_clean.groupby("TotalBaths")["SalePrice"].median().sort_index()
ax.plot(bath_means.index, bath_means.values, "o-", color="#059669", lw=2.5, ms=7)
ax.set_xlabel("Total Bathrooms"); ax.set_ylabel("Median Sale Price ($)")
ax.set_title("TotalBaths vs SalePrice", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig3_feature_eng.png", dpi=130, bbox_inches="tight")
plt.close()
print("[✓] Figure 3 saved — Feature Engineering Insights")

# ─────────────────────────────────────────────
# 6. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[6] Model Training...")
print("-" * 50)

def eval_model(name, model, Xtr, ytr, Xte, yte, is_scaled=False):
    """Train, predict, and return metrics on original price scale."""
    model.fit(Xtr, ytr)
    y_pred_log = model.predict(Xte)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(yte)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    cv   = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2").mean()
    print(f"   {name:<30}  RMSE=${rmse:>9,.0f}  MAE=${mae:>8,.0f}  R²={r2:.4f}  CV-R²={cv:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "CV_R2": cv,
            "y_pred": y_pred, "y_true": y_true}

# Model 1 — Ridge Regression (robust linear)
ridge = Ridge(alpha=10.0, random_state=42)
res_ridge = eval_model("Ridge Regression", ridge, X_train_sc, y_train, X_test_sc, y_test)

# Model 2 — Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_split=4,
                            min_samples_leaf=2, max_features=0.6,
                            n_jobs=-1, random_state=42)
res_rf = eval_model("Random Forest", rf, X_train.values, y_train, X_test.values, y_test)

# Model 3 — Gradient Boosting (GBM — XGBoost equivalent in sklearn)
gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=5, min_samples_leaf=10,
                                  subsample=0.8, max_features=0.7,
                                  random_state=42)
res_gbm = eval_model("Gradient Boosting (GBM)", gbm, X_train.values, y_train, X_test.values, y_test)

print("-" * 50)

results = [res_ridge, res_rf, res_gbm]
best    = min(results, key=lambda x: x["RMSE"])
print(f"\n   🏆 Best model: {best['Model']}  (RMSE ${best['RMSE']:,.0f}, R² {best['R2']:.4f})")

# ─────────────────────────────────────────────
# FIGURE 4 — Model Comparison Bar Charts
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 4 — Model Comparison", fontsize=15, fontweight="bold")

names  = [r["Model"].replace(" (GBM)","") for r in results]
colors = ["#3B82F6","#10B981","#F59E0B"]

for ax, metric, label, fmt in zip(
        axes,
        ["RMSE","MAE","R2"],
        ["RMSE ($)","MAE ($)","R² Score"],
        ["${:,.0f}","${:,.0f}","{:.4f}"]):
    vals = [r[metric] for r in results]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                fmt.format(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.18)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    if metric in ["RMSE","MAE"]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))
    # Highlight best
    best_idx = (vals.index(min(vals)) if metric != "R2" else vals.index(max(vals)))
    bars[best_idx].set_edgecolor("#DC2626")
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig4_model_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n[✓] Figure 4 saved — Model Comparison")

# ─────────────────────────────────────────────
# FIGURE 5 — Actual vs Predicted (Best Model)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Figure 5 — {best['Model']}: Prediction Analysis", fontsize=14, fontweight="bold")

y_true = best["y_true"]
y_pred = best["y_pred"]
residuals = y_true - y_pred

# 5a: Actual vs Predicted
ax = axes[0]
lims = [y_true.min()*0.95, y_true.max()*1.05]
ax.scatter(y_true, y_pred, alpha=0.25, color="#2563EB", s=12)
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Sale Price ($)")
ax.set_ylabel("Predicted Sale Price ($)")
ax.set_title("Actual vs Predicted", fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))
ax.legend()
ax.text(0.05, 0.92, f"R² = {best['R2']:.4f}", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=10)

# 5b: Residuals distribution
ax = axes[1]
ax.hist(residuals, bins=50, color="#7C3AED", edgecolor="white", alpha=0.85)
ax.axvline(0, color="#DC2626", lw=2, ls="--")
ax.set_xlabel("Residual ($)")
ax.set_ylabel("Count")
ax.set_title("Residuals Distribution", fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))
mu, sigma = residuals.mean(), residuals.std()
ax.text(0.04, 0.92, f"μ=${mu:,.0f}\nσ=${sigma:,.0f}", transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8), fontsize=9)

# 5c: Feature importance (if tree-based)
ax = axes[2]
best_model_obj = gbm if "Gradient" in best["Model"] else rf
feat_names  = X_train.columns.tolist()
importances = best_model_obj.feature_importances_
top_n = 15
top_idx = np.argsort(importances)[-top_n:]
top_imp = importances[top_idx]
top_names = [feat_names[i] for i in top_idx]
colors_imp = plt.cm.Blues(np.linspace(0.4, 1.0, top_n))
ax.barh(range(top_n), top_imp, color=colors_imp, edgecolor="white")
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names, fontsize=8)
ax.set_xlabel("Feature Importance")
ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig5_prediction_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print("[✓] Figure 5 saved — Prediction Analysis")

# ─────────────────────────────────────────────
# FIGURE 6 — Neighbourhood & Quality Analysis
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 6 — Market Insights: Neighbourhood & Quality", fontsize=14, fontweight="bold")

ax = axes[0]
nbr_stats = df_clean.groupby("Neighborhood")["SalePrice"].agg(["median","count"]).reset_index()
nbr_stats = nbr_stats.sort_values("median", ascending=True)
c_nbr = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(nbr_stats)))
ax.barh(nbr_stats["Neighborhood"], nbr_stats["median"]/1000, color=c_nbr, edgecolor="white")
ax.set_xlabel("Median Sale Price ($K)")
ax.set_title("Median Price by Neighbourhood", fontweight="bold")
for i, (_, row) in enumerate(nbr_stats.iterrows()):
    ax.text(row["median"]/1000 + 1, i, f"n={int(row['count'])}", va="center", fontsize=8)

ax = axes[1]
qual_box = [df_clean[df_clean["OverallQual"]==q]["SalePrice"].values
            for q in sorted(df_clean["OverallQual"].unique())]
bp = ax.boxplot(qual_box, patch_artist=True, notch=False, showfliers=False)
colors_box = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(qual_box)))
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
ax.set_xticklabels(sorted(df_clean["OverallQual"].unique()))
ax.set_xlabel("Overall Quality (1–10)")
ax.set_ylabel("Sale Price ($)")
ax.set_title("Price Distribution by Quality Grade", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x/1000)}K"))

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig6_market_insights.png", dpi=130, bbox_inches="tight")
plt.close()
print("[✓] Figure 6 saved — Market Insights")

# ─────────────────────────────────────────────
# 7. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL EVALUATION SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame([{
    "Model": r["Model"],
    "RMSE ($)": f"${r['RMSE']:,.0f}",
    "MAE ($)" : f"${r['MAE']:,.0f}",
    "R²"      : f"{r['R2']:.4f}",
    "CV R²"   : f"{r['CV_R2']:.4f}",
    "Best?"   : "★" if r["Model"]==best["Model"] else ""
} for r in results])
print(summary_df.to_string(index=False))
print("\n  Notes:")
print(f"  • Target: log(SalePrice+1), metrics back-transformed to original scale")
print(f"  • Train/test split: 80/20  |  CV: 5-fold on training set")
print(f"  • Feature engineering added: {len(eng_feats)} new features")
print(f"  • Final feature matrix: {X.shape[1]} columns after one-hot encoding")
print("=" * 60)
print("\n[✓] All figures saved to:", SAVE_DIR)