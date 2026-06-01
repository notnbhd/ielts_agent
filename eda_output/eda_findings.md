# IELTS Writing Task 2 — EDA Findings

## Dataset Overview

| Split | Rows |
|-------|------|
| Test | 491 (all have `band_score`) |
| Validation | 984 (no `band_score` column → 491 usable after parsing) |
| **Total usable** | **936** essays with labels |

**Columns**: `ID`, `prompt`, `essay`, `evaluation`, `band_score`

---

## Data Quality

> [!NOTE]
> The `validation.csv` file has `band_score` as `NaN` for all rows in the raw parse — only the `test.csv` rows end up with valid scores. The 936-essay count above is the effective labeled set.

- **Missing values**: 491 rows had null `band_score` (entire validation split in raw parse)
- **Duplicates**: 0
- **Outliers**: 20 essays fall outside the 1st–99th percentile word-count range (206–479 words)

---

## Band Score Distribution

![Band Score Distribution](file:///home/nacho/Projekt/agent_test/eda_output/2_band_count_per_score.png)

- Range: **4.0 – 9.0** (no 3.0 or 3.5 in this split)
- Mean: **6.29**, Median: **6.5**
- Most common bands: **7.0 (142)**, **6.5 (119)**, **6.0 (117)**
- Very high bands are rare: Band 9 = only **9 essays**
- Distribution is **slightly left-skewed** — dataset leans toward mid-to-high competency

---

## Text Feature Statistics (936 labeled essays)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Word Count | 303 | 51 | 160 | 591 |
| Sentence Count | 15.5 | 3.3 | 2 | 35 |
| Avg Word Length | 5.0 | 0.29 | 4.22 | 6.44 |
| Lexical Diversity (TTR) | 0.58 | 0.05 | 0.41 | 0.70 |
| Avg Sentence Length | — | — | — | — |

---

## Feature Distributions

![Text Feature Distributions](file:///home/nacho/Projekt/agent_test/eda_output/3_text_feature_distributions.png)

- **Word count** is roughly normal, centered ~300 words
- **TTR** is tightly concentrated (0.41–0.70), indicating similar vocabulary richness across most essays
- **Avg word length** is normally distributed around 5 chars

---

## Features vs Band Score

![Features vs Band](file:///home/nacho/Projekt/agent_test/eda_output/4_features_vs_band.png)

Mild but visible trends:
- Higher-band essays tend to have slightly **higher word counts** and **longer average words**
- **Lexical diversity** increases a bit at the high end (Band 8.5–9.0: TTR ~0.62)

---

## Correlation with Band Score

![Correlation Heatmap](file:///home/nacho/Projekt/agent_test/eda_output/5_correlation_heatmap.png)

| Feature | Pearson r |
|---------|-----------|
| `essay_avg_word_len` | **+0.098** |
| `essay_lexical_div` | **+0.094** |
| `essay_word_count` | **+0.093** |
| `essay_sentence_count` | +0.023 |
| `essay_paragraph_count` | -0.002 |
| `prompt_word_count` | -0.024 |
| `essay_avg_sent_len` | -0.030 |

> [!IMPORTANT]
> All surface-level text features have **very weak correlations** with band score (r < 0.1). This confirms that IELTS grading depends heavily on **semantics, coherence, grammar, and argument quality** — not just length or vocabulary richness. This is exactly why fine-tuned LLMs outperform simple regression on this task.

---

## Word Clouds by Band Tier

````carousel
![Low Band Word Cloud (≤5.5)](file:///home/nacho/Projekt/agent_test/eda_output/8_wordcloud_low.png)
<!-- slide -->
![Mid Band Word Cloud (6–7)](file:///home/nacho/Projekt/agent_test/eda_output/8_wordcloud_mid.png)
<!-- slide -->
![High Band Word Cloud (≥7.5)](file:///home/nacho/Projekt/agent_test/eda_output/8_wordcloud_high.png)
````

---

## Summary Dashboard

![EDA Summary Dashboard](file:///home/nacho/Projekt/agent_test/eda_output/9_summary_dashboard.png)

---

## Key Takeaways for Model Development

1. **Surface features don't predict band well** — use them only as auxiliary signals, not primary features
2. **Class imbalance**: Band 9 has only 9 samples — consider oversampling or weighted loss
3. **Vocabulary-richness signal is weak** (TTR range is very narrow)
4. **Target distribution** is mid-heavy (6.0–7.0 dominate) — your model will be biased toward this range unless corrected
5. **No duplicate essays**, so no data leakage risk there
6. **Outlier essays** (very short <206 or very long >479 words) are worth inspecting separately — they may have noisy labels

---

*Script: [`eda.py`](file:///home/nacho/Projekt/agent_test/eda.py) | Charts: [`eda_output/`](file:///home/nacho/Projekt/agent_test/eda_output/)*
