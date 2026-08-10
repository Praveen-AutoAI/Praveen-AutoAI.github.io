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

**Pearson correlation** measures the strength and direction of the **linear relationship** between a feature \(X\) and the target variable \(Y\).

$$
r_{XY} =
\frac{
\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})
}{
\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2}
\sqrt{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}
}
$$

## Intuitive Example

When the recorded current increases, the measured temperature may also increase:

- **Current** ↑  
- **Temperature** ↑  

A strong linear trend between the two variables produces a high absolute Pearson correlation score.

## Why Engineers Love It

- **Fast** to calculate across a large number of recorded variables
- **Easy to interpret**
- Works well for **first-pass feature screening**
- Indicates both the **strength** and **direction** of a linear relationship
- Helps quickly reduce a large feature set to potentially relevant variables

## Why It Can Fail

Pearson correlation fundamentally assumes that an important relationship can be represented by a **straight line**:

> **Important relationship = Straight-line relationship**

However, physical systems rarely behave in a completely linear manner.

Examples of nonlinear behaviour include:

- Battery degradation
- Thermal runaway
- Motor efficiency
- Fluid-flow dynamics
- Component saturation
- Threshold-based control responses

Therefore, a variable may have a significant influence on the target while still receiving a low Pearson correlation score. Pearson correlation should consequently be used as an **initial diagnostic and screening tool**, rather than as the only method for identifying high-impact variables.

