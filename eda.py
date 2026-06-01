"""
eda.py — Exploratory Data Analysis for IELTS Writing Task 2 Dataset
Run: uv run python eda.py
Outputs: eda_output/ directory with all charts + a summary printed to stdout.
"""

import os
import re
import string
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = sns.color_palette("mako", 12)
BAND_ORDER = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]

sns.set_theme(style="darkgrid", palette="mako", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})

# ─────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("  IELTS Writing Task 2 — Exploratory Data Analysis")
print("=" * 60)

test_df  = pd.read_csv("writting_task2_dataset/test.csv")
val_df   = pd.read_csv("writting_task2_dataset/validation.csv")

# merge splits for holistic EDA; keep a 'split' column
test_df["split"] = "test"
val_df["split"]  = "validation"
df = pd.concat([test_df, val_df], ignore_index=True)

print(f"\n[1] Dataset loaded")
print(f"    test rows      : {len(test_df):,}")
print(f"    validation rows: {len(val_df):,}")
print(f"    total rows     : {len(df):,}")
print(f"    columns        : {list(df.columns)}")

# ─────────────────────────────────────────────
# 2.  Basic data quality
# ─────────────────────────────────────────────
print("\n[2] Data Quality")
print(df.dtypes.to_string())
print("\nMissing values:")
print(df.isnull().sum().to_string())
print("\nDuplicate rows:", df.duplicated().sum())

df["band_score"] = pd.to_numeric(df["band_score"], errors="coerce")
df.dropna(subset=["band_score", "essay"], inplace=True)

# ─────────────────────────────────────────────
# 3.  Feature engineering
# ─────────────────────────────────────────────
def count_words(text: str) -> int:
    return len(str(text).split())

def count_sentences(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+", str(text).strip())))

def count_paragraphs(text: str) -> int:
    return max(1, len([p for p in str(text).split("\n\n") if p.strip()]))

def avg_word_length(text: str) -> float:
    words = [w.strip(string.punctuation) for w in str(text).split() if w.strip(string.punctuation)]
    return np.mean([len(w) for w in words]) if words else 0.0

def lexical_diversity(text: str) -> float:
    tokens = str(text).lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def avg_sentence_length(text: str) -> float:
    sentences = re.split(r"[.!?]+", str(text).strip())
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return np.mean(lengths) if lengths else 0.0

print("\n[3] Engineering text features...")
df["essay_word_count"]      = df["essay"].apply(count_words)
df["essay_sentence_count"]  = df["essay"].apply(count_sentences)
df["essay_paragraph_count"] = df["essay"].apply(count_paragraphs)
df["essay_avg_word_len"]    = df["essay"].apply(avg_word_length)
df["essay_lexical_div"]     = df["essay"].apply(lexical_diversity)
df["essay_avg_sent_len"]    = df["essay"].apply(avg_sentence_length)
df["essay_char_count"]      = df["essay"].apply(lambda x: len(str(x)))

df["prompt_word_count"] = df["prompt"].apply(count_words)

print(df[["essay_word_count", "essay_sentence_count", "essay_avg_word_len",
          "essay_lexical_div", "band_score"]].describe().round(2).to_string())

# ─────────────────────────────────────────────
# 4.  Band-score distribution
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Band Score Distribution", fontsize=15, fontweight="bold")

# 4a. Overall histogram
ax = axes[0]
df["band_score"].hist(bins=20, color=PALETTE[4], edgecolor="white", ax=ax)
ax.set_xlabel("Band Score")
ax.set_ylabel("Count")
ax.set_title("All splits combined")
ax.axvline(df["band_score"].mean(), color="#f97316", linestyle="--", label=f"Mean={df['band_score'].mean():.2f}")
ax.legend()

# 4b. By split
ax = axes[1]
for split, grp in df.groupby("split"):
    grp["band_score"].hist(bins=20, alpha=0.65, ax=ax, label=split)
ax.set_xlabel("Band Score")
ax.set_ylabel("Count")
ax.set_title("By split")
ax.legend()

plt.tight_layout()
path = f"{OUTPUT_DIR}/1_band_score_distribution.png"
plt.savefig(path); plt.close()
print(f"\n[4] Saved → {path}")

# 4c. Count per discrete band
fig, ax = plt.subplots(figsize=(12, 5))
counts = df["band_score"].value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[3], edgecolor="white", width=0.65)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, str(int(b.get_height())),
            ha="center", va="bottom", fontsize=9, color="white")
ax.set_xlabel("Band Score")
ax.set_ylabel("Number of Essays")
ax.set_title("Essay Count per Band Score")
plt.tight_layout()
path = f"{OUTPUT_DIR}/2_band_count_per_score.png"
plt.savefig(path); plt.close()
print(f"[4] Saved → {path}")

# ─────────────────────────────────────────────
# 5.  Text feature distributions
# ─────────────────────────────────────────────
TEXT_FEATURES = {
    "essay_word_count":      "Word Count",
    "essay_sentence_count":  "Sentence Count",
    "essay_avg_word_len":    "Avg Word Length (chars)",
    "essay_lexical_div":     "Lexical Diversity (TTR)",
    "essay_avg_sent_len":    "Avg Sentence Length (words)",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Essay Text Feature Distributions", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, (col, label) in enumerate(TEXT_FEATURES.items()):
    ax = axes[i]
    ax.hist(df[col], bins=40, color=PALETTE[i + 1], edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(), color="#f97316", linestyle="--", label=f"μ={df[col].mean():.1f}")
    ax.set_title(label); ax.set_xlabel(label); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# hide the unused 6th subplot
axes[-1].set_visible(False)
plt.tight_layout()
path = f"{OUTPUT_DIR}/3_text_feature_distributions.png"
plt.savefig(path); plt.close()
print(f"[5] Saved → {path}")

# ─────────────────────────────────────────────
# 6.  Features vs Band Score (box plots)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Text Features vs Band Score", fontsize=15, fontweight="bold")
axes = axes.flatten()

for i, (col, label) in enumerate(TEXT_FEATURES.items()):
    ax = axes[i]
    data_by_band = [df.loc[df["band_score"] == b, col].values for b in BAND_ORDER if b in df["band_score"].values]
    present_bands = [b for b in BAND_ORDER if b in df["band_score"].values]
    bp = ax.boxplot(data_by_band, labels=[str(b) for b in present_bands],
                    patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xlabel("Band Score"); ax.set_ylabel(label)
    ax.set_title(f"{label} by Band Score")

axes[-1].set_visible(False)
plt.tight_layout()
path = f"{OUTPUT_DIR}/4_features_vs_band.png"
plt.savefig(path); plt.close()
print(f"[6] Saved → {path}")

# ─────────────────────────────────────────────
# 7.  Correlation heatmap
# ─────────────────────────────────────────────
num_cols = ["band_score", "essay_word_count", "essay_sentence_count",
            "essay_paragraph_count", "essay_avg_word_len",
            "essay_lexical_div", "essay_avg_sent_len", "prompt_word_count"]

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="mako",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
path = f"{OUTPUT_DIR}/5_correlation_heatmap.png"
plt.savefig(path); plt.close()
print(f"[7] Saved → {path}")

# Print top correlations with band_score
print("\n    Correlations with band_score:")
print(corr["band_score"].drop("band_score").sort_values(ascending=False).round(3).to_string())

# ─────────────────────────────────────────────
# 8.  Word count vs band score (scatter + trend)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df["essay_word_count"], df["band_score"],
                c=df["band_score"], cmap="mako", alpha=0.4, s=18, linewidths=0)
plt.colorbar(sc, ax=ax, label="Band Score")

# trend line
z = np.polyfit(df["essay_word_count"], df["band_score"], 1)
p = np.poly1d(z)
xs = np.linspace(df["essay_word_count"].min(), df["essay_word_count"].max(), 200)
ax.plot(xs, p(xs), color="#f97316", lw=2, label=f"Trend (r={df[['essay_word_count','band_score']].corr().iloc[0,1]:.2f})")
ax.set_xlabel("Essay Word Count")
ax.set_ylabel("Band Score")
ax.set_title("Essay Length vs Band Score")
ax.legend()
plt.tight_layout()
path = f"{OUTPUT_DIR}/6_word_count_vs_band.png"
plt.savefig(path); plt.close()
print(f"[8] Saved → {path}")

# ─────────────────────────────────────────────
# 9.  Lexical diversity vs band score
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df["essay_lexical_div"], df["band_score"],
                c=df["band_score"], cmap="mako", alpha=0.4, s=18, linewidths=0)
plt.colorbar(sc, ax=ax, label="Band Score")
z = np.polyfit(df["essay_lexical_div"], df["band_score"], 1)
p = np.poly1d(z)
xs = np.linspace(df["essay_lexical_div"].min(), df["essay_lexical_div"].max(), 200)
r  = df[["essay_lexical_div", "band_score"]].corr().iloc[0, 1]
ax.plot(xs, p(xs), color="#f97316", lw=2, label=f"Trend (r={r:.2f})")
ax.set_xlabel("Type-Token Ratio (Lexical Diversity)")
ax.set_ylabel("Band Score")
ax.set_title("Lexical Diversity vs Band Score")
ax.legend()
plt.tight_layout()
path = f"{OUTPUT_DIR}/7_lexical_div_vs_band.png"
plt.savefig(path); plt.close()
print(f"[9] Saved → {path}")

# ─────────────────────────────────────────────
# 10.  Word clouds per band tier
# ─────────────────────────────────────────────
STOPWORDS = set("""the a an and or but in on at to of for is are was were be been
    being have has had do does did will would could should may might shall
    with this that these those it its i you he she we they them their our
    my your his her its from by as about into through during before after
    above below between out off over under again further then once here there
    when where why how all both each few more most other some such no nor
    not only same so than too very just because if which who whom""".split())

def make_wordcloud(texts, title, filepath, colormap="mako"):
    combined = " ".join(str(t) for t in texts)
    tokens   = [w.lower().strip(string.punctuation)
                for w in combined.split()
                if w.lower().strip(string.punctuation) not in STOPWORDS
                and len(w.strip(string.punctuation)) > 2]
    freq = Counter(tokens)
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap=colormap, max_words=150).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath); plt.close()

# Tier definition: Low (≤5.5), Mid (6.0-7.0), High (≥7.5)
tiers = {
    "Low Band (≤5.5)":   df[df["band_score"] <= 5.5]["essay"],
    "Mid Band (6–7)":    df[(df["band_score"] >= 6.0) & (df["band_score"] <= 7.0)]["essay"],
    "High Band (≥7.5)":  df[df["band_score"] >= 7.5]["essay"],
}

colormaps = ["YlOrRd", "Blues", "Greens"]
for (label, texts), cmap in zip(tiers.items(), colormaps):
    path = f"{OUTPUT_DIR}/8_wordcloud_{label.split()[0].lower()}.png"
    make_wordcloud(texts, f"Word Cloud — {label}", path, colormap=cmap)
    print(f"[10] Saved → {path}")

# ─────────────────────────────────────────────
# 11.  Band score stats table
# ─────────────────────────────────────────────
print("\n[11] Per-band statistics:")
per_band = df.groupby("band_score").agg(
    count=("essay_word_count", "count"),
    avg_words=("essay_word_count", "mean"),
    avg_sentences=("essay_sentence_count", "mean"),
    avg_word_len=("essay_avg_word_len", "mean"),
    avg_lexical_div=("essay_lexical_div", "mean"),
).round(2)
print(per_band.to_string())

# ─────────────────────────────────────────────
# 12.  Outlier analysis
# ─────────────────────────────────────────────
print("\n[12] Outlier analysis (essays with extreme word counts):")
q_low  = df["essay_word_count"].quantile(0.01)
q_high = df["essay_word_count"].quantile(0.99)
outliers = df[(df["essay_word_count"] < q_low) | (df["essay_word_count"] > q_high)]
print(f"    <1st pct ({q_low:.0f} words) or >99th pct ({q_high:.0f} words): {len(outliers)} essays")
print(outliers[["ID", "band_score", "essay_word_count"]].head(10).to_string())

# ─────────────────────────────────────────────
# 13.  Final summary figure
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("IELTS Writing Task 2 — EDA Summary", fontsize=17, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 13a. Band distribution
ax1 = fig.add_subplot(gs[0, :2])
df["band_score"].value_counts().sort_index().plot(kind="bar", ax=ax1, color=PALETTE[3], edgecolor="white", width=0.75)
ax1.set_title("Band Score Frequency"); ax1.set_xlabel("Band"); ax1.set_ylabel("Count")

# 13b. Word count KDE by split
ax2 = fig.add_subplot(gs[0, 2])
for split, grp in df.groupby("split"):
    grp["essay_word_count"].plot(kind="kde", ax=ax2, label=split)
ax2.set_title("Word Count KDE by Split"); ax2.legend()

# 13c. Correlation bar chart
ax3 = fig.add_subplot(gs[1, :])
corr_vals = corr["band_score"].drop("band_score").sort_values()
colors = ["#f97316" if v > 0 else "#6366f1" for v in corr_vals]
bars = ax3.barh(corr_vals.index, corr_vals.values, color=colors, edgecolor="white", height=0.6)
ax3.axvline(0, color="white", linewidth=0.8)
ax3.set_title("Feature Correlation with Band Score")
ax3.set_xlabel("Pearson r")
for bar, val in zip(bars, corr_vals.values):
    ax3.text(val + 0.005 * np.sign(val), bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", ha="left" if val > 0 else "right", fontsize=9)

plt.tight_layout()
path = f"{OUTPUT_DIR}/9_summary_dashboard.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"\n[13] Summary dashboard saved → {path}")

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  ✓ EDA complete. All charts in: {OUTPUT_DIR}/")
print("=" * 60)
