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

#### Pearson Correlation: The Engineer's First Diagnostic Tool

# Pearson Correlation

## 1. What is it?

Pearson Correlation is one of the simplest and most widely used feature-ranking methods. It quantifies the strength and direction of a linear relationship between a predictor variable and a target variable.

The Pearson coefficient ranges from:

- +1 : Perfect positive relationship
- 0 : No linear relationship
- -1 : Perfect negative relationship

Features with larger absolute correlation values are considered more influential on the target.

---

## 2. Mathematical Foundation

The Pearson correlation coefficient is defined as:

$$
r_{XY} =
\frac{
\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)
}
{
\sqrt{\sum_{i=1}^{n}(X_i-\bar X)^2}
\sqrt{\sum_{i=1}^{n}(Y_i-\bar Y)^2}
}
$$

where:

- X = feature
- Y = target variable
- n = number of observations
- rXY = correlation coefficient

---

## 3. How It Works

1. Select a target variable.
2. Calculate the Pearson correlation between every feature and the target.
3. Rank features based on absolute correlation magnitude.
4. Retain the highest-ranking features for further analysis.

Example:

- Feature Count = 1500
- Ranked Features = 1500
- Selected Features = Top 50-100

---

## 4. Engineering Intuition

Consider an electric vehicle thermal test.

Target:

- Battery Cell Temperature

Available signals:

- Motor Current
- Coolant Flow Rate
- Pump Speed
- Ambient Temperature
- Vehicle Speed

If motor current increases and battery temperature increases almost proportionally, Pearson correlation will assign a high positive score.

For example:

Current ↑ → Temperature ↑

Such behavior indicates a strong linear relationship and therefore a high ranking.

---

## 5. Strengths

- Extremely fast computationally
- Easy to understand and explain
- Scales well to thousands of signals
- Useful as an initial screening method
- Provides directional information (positive or negative influence)

---

## 6. Weaknesses

- Captures only linear relationships
- Sensitive to outliers
- Misses nonlinear dependencies
- Cannot identify feature interactions
- Can underestimate physically important variables

For example, battery cooling systems often exhibit threshold behavior. Such relationships may be highly influential but poorly captured by Pearson correlation.

---

## 7. Relevance to Automotive Feature Ranking

Modern vehicle tests routinely generate more than 1000 recorded signals.

Pearson correlation is particularly useful as a first-level filter because it:

- Quickly eliminates irrelevant variables
- Highlights dominant physical relationships
- Reduces computational burden for more advanced methods
- Accelerates calibration investigations
- Supports root-cause analysis during validation

For feature-reduction pipelines, Pearson correlation is often used before applying machine-learning-based methods such as Random Forest, Boruta, or mRMR.

---

## 8. Practical Interpretation of Scores

| Absolute Correlation | Interpretation |
|---------------------|---------------|
| > 0.80 | Very Strong Relationship |
| 0.60 - 0.80 | Strong Relationship |
| 0.40 - 0.60 | Moderate Relationship |
| 0.20 - 0.40 | Weak Relationship |
| < 0.20 | Very Weak Relationship |

Note: Thresholds should be treated as guidelines and may vary depending on the application.

---

## 9. When Should Engineers Use It?

Recommended for:

- Initial feature screening
- Large signal databases
- Exploratory analysis
- Validation studies
- Quick health checks of logged datasets

---

## 10. When Should Engineers Avoid It?

Avoid using Pearson correlation as the sole ranking method when:

- Relationships are nonlinear
- Strong feature interactions exist
- Threshold effects are expected
- Battery aging studies are involved
- Thermal systems exhibit saturation behavior

---

## 11. Final Verdict

Pearson correlation is best viewed as the engineer's first diagnostic tool rather than a complete feature-ranking solution. It offers exceptional speed and interpretability for large vehicle-testing datasets and serves as an effective first-stage filter before applying more sophisticated techniques capable of capturing nonlinear relationships and feature interactions.
Therefore, a variable may have a significant influence on the target while still receiving a low Pearson correlation score. Pearson correlation should consequently be used as an **initial diagnostic and screening tool**, rather than as the only method for identifying high-impact variables.

