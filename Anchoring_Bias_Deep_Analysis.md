# Anchoring Bias Experiment - Deep Analysis Report

**Dataset:** 368 observations from 14 participants across 8 matrices  
**Date:** December 2025

---

## Executive Summary

**⚠️ Key Finding: The overall anchoring effect is NOT statistically significant.**

| Metric | Value |
|--------|-------|
| Overall Effect | 0.53 squares |
| Cohen's d | 0.026 (negligible) |
| p-value | 0.804 |
| 95% CI | [-3.68, 4.74] |

However, this masks dramatic **heterogeneity across matrices** - some matrices produced strong anchoring effects while others produced *reverse* effects.

---

## 1. Matrix-Level Effectiveness

Not all matrices are created equal. Here's how each performed:

| Matrix | True Count | Cohen's d | Interpretation |
|--------|-----------|-----------|----------------|
| **#5** | 58 | **0.57** | ✅ Medium effect - **WORKS** |
| #1 | 57 | 0.09 | ❌ No effect |
| #4 | 59 | 0.07 | ❌ No effect |
| #2 | 43 | 0.07 | ❌ No effect |
| #3 | 46 | 0.03 | ❌ No effect |
| #6 | 36 | -0.11 | ⚠️ Reverse effect |
| #7 | 48 | -0.17 | ⚠️ Reverse effect |
| **#8** | 54 | **-0.25** | ⚠️ Small reverse effect |

**Insight:** Only Matrix #5 demonstrates meaningful anchoring. Matrices #6, #7, #8 show *negative* effects (high anchors → lower estimates). This could indicate:
- Design issues with those specific grids
- Counter-anchoring (participants correcting away from obvious anchors)
- Random noise given small per-matrix samples

---

## 2. Learning/Fatigue Effects

**Question:** Do participants "catch on" to the manipulation over 16 rounds?

**Answer:** No significant trend detected.

- Slope: -0.06 squares/round (essentially flat)
- R² = 0.01
- p = 0.71

The anchoring effect (or lack thereof) remains consistent from round 1 to round 16.

---

## 3. High vs Low Anchor Asymmetry

**Question:** Are participants more susceptible to high anchors or low anchors?

| Condition | Mean Error |
|-----------|------------|
| High anchor (+15%) | -1.67 squares |
| Low anchor (-15%) | -2.21 squares |

Both conditions produce **underestimation**, with no significant asymmetry (t=-0.79, p=0.43). This is unusual - typically high anchors have stronger effects.

---

## 4. Individual Participant Profiles

Participants varied dramatically in susceptibility:

| Category | Count | Participants |
|----------|-------|--------------|
| **Resistant** (|pull| < 2) | 9 | CGZ, Patricia GZ, Nezmaux, laplusbellesoeurdumonde, Ka, Neymar, ClementGzl, Arnaud, Anastasia |
| **Moderate** (2-5) | 5 | Yass (+3.19), Dylan (+2.88), Cecile (+2.06), Baptiste (-2.31), learmbd (-2.31) |
| **Strong** (≥5) | 0 | None |

**Notable:** Baptiste and learmbd showed *reverse* anchoring - they estimated lower when shown high anchors.

---

## 5. Systematic Estimation Bias (Independent of Anchoring)

**Overall mean error: -1.94 squares (p < 0.001)**

Participants systematically **underestimate** blue square counts, independent of anchoring. This is a classic finding in visual estimation tasks.

**Extreme cases:**
- Patricia GZ: -17.69 (severe underestimation)
- laplusbellesoeurdumonde: -12.06 
- learmbd: +5.56 (overestimation)

---

## 6. Digit Preference Analysis

Participants strongly prefer round numbers:

| Last Digit | Observed | Expected (10%) |
|------------|----------|----------------|
| 0 | 18.5% | 10% |
| 5 | 20.4% | 10% |
| **Total 0 or 5** | **38.9%** | **20%** |

This is double the expected rate - participants round their estimates heavily.

**Correlation:** Round-number preferrers showed slightly *less* anchoring bias (r = -0.33).

---

## 7. Task Difficulty by Matrix

Some matrices were inherently harder to estimate (regardless of anchoring):

| Matrix | True Count | MAE | Bias |
|--------|-----------|-----|------|
| #1 | 57 | 7.04 | -10.0 (underest.) |
| #2 | 43 | 8.15 | +6.1 (overest.) |
| #6 | 36 | 10.70 | +8.8 (overest.) |

**Finding:** Higher true counts correlate with higher errors (r = 0.13, p = 0.01).

---

## 8. Order Effects

For each matrix, participants saw both a high-anchor and low-anchor version. Does order matter?

| Sequence | 1st Round Pull | 2nd Round Pull |
|----------|---------------|----------------|
| High→Low | -0.42 | +2.67 |
| Low→High | +1.74 | -2.96 |

**Interesting pattern:** The second exposure tends to show a correction *away* from the first anchor. This could explain the weak overall effect - memory of the previous estimate interferes.

---

## 9. Matrix Quality Score

Combining anchoring effectiveness and task difficulty:

**Quality = Cohen's d - (MAE / 20)**

| Rank | Matrix | Score | Recommendation |
|------|--------|-------|----------------|
| 1 | **#5** | 0.13 | ✅ Keep - best performer |
| 2 | #1 | -0.26 | Consider |
| 3 | #2 | -0.34 | Marginal |
| 4 | #4 | -0.35 | Marginal |
| 5 | #3 | -0.44 | Weak |
| 6 | #7 | -0.61 | ❌ Redesign |
| 7 | #6 | -0.65 | ❌ Redesign |
| 8 | **#8** | -0.75 | ❌ Remove |

---

## Recommendations for Future Iterations

1. **Investigate Matrix #5:** Why does it work when others don't? Compare visual characteristics.

2. **Remove or redesign matrices #6, #7, #8:** They show reverse effects and hurt overall power.

3. **Increase anchor strength:** ±15% may be too subtle. Consider ±20-25%.

4. **Address order effects:** Consider showing each matrix only once (double participants instead).

5. **Screen for systematic bias:** Some participants (Patricia GZ) may have perceptual issues that mask anchoring.

6. **Larger sample:** 14 participants is underpowered for detecting small effects.

---

## Statistical Summary Table

| Analysis | Key Statistic | p-value | Significant? |
|----------|--------------|---------|--------------|
| Overall anchoring | d = 0.026 | 0.804 | ❌ No |
| Best matrix (#5) | d = 0.57 | 0.061 | Marginal |
| Learning trend | slope = -0.06 | 0.711 | ❌ No |
| Asymmetry | Δ = 0.53 | 0.432 | ❌ No |
| Systematic underest. | M = -1.94 | 0.001 | ✅ Yes |
| Digit preference | 38.9% vs 20% | — | ✅ Yes |

---

*Analysis generated from 368 observations across 14 participants*
