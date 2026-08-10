---
layout: post
title: "Finding Signals from Noise : Feature Ranking to Identify the Cause and Effect"
date: 2026-08-10
categories: [Machine Learning, Engineering]
tags: [PINN, Physics, Deep Learning, AI]
math: true
---

### Background

**Vehicle development and validation involve the continuous acquisition** of a large number of signals representing performance, thermal, electrical, and operating conditions. In a typical test campaign, approximately 1,500 recorded variables or labels may be available, making it difficult to determine which parameters have the greatest influence on a specific target response. To address this challenge, a statistic based **feature-ranking methodology was developed to systematically identify and prioritize the variables** most relevant to the target variable. The method evaluates the strength and consistency of each feature’s relationship with the target, while also considering redundancy, data quality, and potential nonlinear effects. Based on the resulting importance scores, the complete feature set is reduced to a focused list of approximately 50 to 100 high-impact or potentially influential variables. This prioritized list enables engineers to concentrate their analysis on the parameters most likely to affect vehicle behavior, thereby improving the efficiency and interpretability of the development process. Identifying these critical variables **supports informed calibration decisions, accelerates root-cause investigation, and reduces the effort required to analyze large, complex datasets**. Ultimately, the methodology provides a data-driven basis for calibrating vehicle systems to satisfy performance targets and thermal and electrical requirements while improving testing efficiency and overall development robustness.

This method sounds like a simple correlation study, but developing a pipeline that successfully works for various kinds of experimental data(vehicle/powertrain/component level testing, etc) and able to rank high impact variables consistently across the applications are real challenge. The **use-cases are immense and the value it add in terms of the Quality/Cost/Time (QCT) benefits is incredible**.

### Objective

Most engineering datasets contain hundreds of sensors, calculated signals, timestamps, operational flags, and derived parameters. Many of these variables are redundant, some are pure noise, and a few contain the information that ultimately affects the target variable. **Determine the variables that are potentially impact the target variable**. 
> Can a pure statistically driven method can do? To what extent?

### Introduction to Feature Ranking(/Importance/Selection) Methods

Feature selection/importance area is pretty vast and consists of numerous methods, I suggest to go through the overview and general methods of feature engineering/selection from the below link. 
#### Before jumping deep into the ocean, I let you surf a little
* [Feature Selection — Exhaustive Overview (Analytics Vidhya)](https://medium.com/analytics-vidhya/feature-selection-extended-overview-b58f1d524c1c)
* [Feature Selection Techniques in Machine Learning (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/)

**Landscape of Feature Selection/Importance**
![Landscape of Feature Selection/Importance](/assets/images/feature_Engineering_methods.jpg)

> Yes, the landscape is vast and still there are lot of methods. Generally we can group them into 3 class:
> Filter Method: Evaluates each feature independently with respect to the target variable
> Wrapper Method: Evaluates different combinations of features by measuring their impact on model performance
> Embedded Method: Performs feature selection during the model training process
> Check out the links to understand more about the classes

Based on my experimentation with many methods and I found handfull of methods that are really robust that can help you identify the signals trapped in the noise. Let's begin!!!

# Pearson Correlation

## 1. What is it?

Pearson Correlation is a statistical technique used to measure the strength and direction of a **linear relationship** between a feature and a target variable. The correlation coefficient ranges from **-1 to +1**, where positive values indicate that the target increases as the feature increases, negative values indicate an inverse relationship, and values close to zero indicate little or no linear dependency.

For feature ranking, variables are ranked based on the absolute correlation coefficient, with larger values indicating stronger influence on the target.

---

## 2. Mathematical Foundation

\[
r_{XY} =
\frac{
\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)
}
{
\sqrt{\sum_{i=1}^{n}(X_i-\bar X)^2}
\sqrt{\sum_{i=1}^{n}(Y_i-\bar Y)^2}
}
\]

Where:

- \(X\) = Feature variable
- \(Y\) = Target variable
- \(\bar X\) = Mean of feature values
- \(\bar Y\) = Mean of target values
- \(r_{XY}\) = Pearson correlation coefficient

The coefficient quantifies how strongly the feature and target vary together relative to their individual variations.

---

## 3. Strengths and Weaknesses

| Strengths | Weaknesses |
|------------|------------|
| Fast and computationally efficient | Detects only linear relationships |
| Easy to understand and interpret | Misses nonlinear dependencies |
| Suitable for large datasets | Sensitive to outliers |
| Provides direction of influence | Cannot capture feature interactions |
| Effective for preliminary screening | Produces redundant rankings for correlated features |
| Widely accepted and easy to communicate | May overlook physically important nonlinear variables |

---

## 4. Relevance to Automotive Feature Ranking

Modern vehicle testing programs typically record more than 1,500 signals from powertrain, battery, thermal, electrical, and control systems. Pearson Correlation serves as an effective first-pass feature ranking tool by identifying variables that exhibit strong linear relationships with a target parameter.

Typical applications include:

- Battery temperature prediction
- Energy consumption analysis
- Motor torque investigation
- Thermal system calibration
- Root-cause analysis

Its primary advantage is the ability to rapidly reduce thousands of signals to a manageable set of potentially influential variables. However, automotive systems frequently exhibit nonlinear behaviors such as thermal saturation, battery aging effects, efficiency maps, and control logic thresholds. As a result, Pearson Correlation should be viewed as an initial screening method rather than a standalone feature-selection technique.

It is most effective when used as part of a broader feature-ranking framework alongside nonlinear methods such as Spearman Correlation, ReliefF, Random Forest Importance, mRMR, and Boruta.

# Spearman Correlation

## 1. What is it?

Spearman Correlation is a non-parametric statistical technique that measures the strength and direction of a **monotonic relationship** between a feature and a target variable. Unlike Pearson Correlation, it operates on ranked data and can identify relationships that consistently increase or decrease, even when the relationship is nonlinear.

The Spearman coefficient ranges from **-1 to +1**, with larger absolute values indicating stronger monotonic relationships.

---

## 2. Mathematical Foundation

\[
\rho =
1 - \frac{6\sum d_i^2}{n(n^2-1)}
\]

Where:

- \(\rho\) = Spearman correlation coefficient
- \(d_i\) = Difference between the ranks of corresponding observations
- \(n\) = Number of observations

Instead of comparing actual values, Spearman evaluates how consistently the rankings of a feature and target move together.

---

## 3. Strengths and Weaknesses

| Strengths | Weaknesses |
|------------|------------|
| Detects linear and monotonic nonlinear relationships | Cannot detect complex non-monotonic relationships |
| Less sensitive to outliers | Loses some information through ranking |
| Suitable for skewed and noisy data | Cannot model feature interactions |
| Computationally efficient | May miss operating-region-dependent behavior |
| Captures dependencies missed by Pearson | Correlated variables may receive similar scores |
| Well suited for real-world engineering datasets | Less directly interpretable than Pearson |

---

## 4. Relevance to Automotive Feature Ranking

Many automotive systems exhibit nonlinear but monotonic behavior. Examples include battery temperature versus current, inverter temperature versus load, battery voltage versus state of charge, and aerodynamic losses versus vehicle speed. While these relationships may not be linear, they often maintain a consistent increasing or decreasing trend.

Spearman Correlation is particularly valuable because it can identify such relationships more effectively than Pearson Correlation. By focusing on ranked trends rather than actual values, it also becomes more robust to measurement noise and outliers, which are commonly encountered during vehicle testing.

This makes Spearman Correlation highly suitable for:

- Battery performance studies
- Thermal system analysis
- Powertrain efficiency investigations
- Calibration support activities
- Validation and root-cause analysis

For large-scale vehicle datasets containing thousands of logged signals, Spearman Correlation provides a reliable screening mechanism for identifying high-impact variables. It often serves as an intermediate feature-ranking method between simple linear analysis and more advanced machine-learning-based approaches such as Random Forest Importance, Permutation Importance, ReliefF, mRMR, and Boruta.
