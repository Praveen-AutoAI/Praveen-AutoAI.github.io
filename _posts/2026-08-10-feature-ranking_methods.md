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
## **Methods Used**
![Methods_used](/assets/images/Classification_.jpg)

## **#1 Pearson's Correlation**
Pearson Correlation is a statistical technique used to measure the strength and direction of a **linear relationship** between a feature and a target variable. The correlation coefficient ranges from **-1 to +1**, where positive values indicate that the target increases as the feature increases, negative values indicate an inverse relationship, and values close to zero indicate little or no linear dependency.

![Pearsons_Correlation](/assets/images/pearson.jpg)

>**Relevance to Automotive Feature Ranking**

- Rapidly screens thousands of recorded vehicle signals to identify variables with strong linear relationships to the target.
- Helps reduce large datasets (e.g., 1,500+ signals) to a manageable set of candidate features.
- Supports calibration activities by highlighting parameters that directly influence vehicle performance.
- Useful for root-cause investigations involving thermal, electrical, and powertrain systems.
- Computationally efficient and suitable for automated feature-ranking pipelines.
- Best used as an initial filtering technique before applying advanced nonlinear feature-ranking methods.

## **#2 Spearman's Correlation**
Spearman Correlation is a non-parametric statistical technique that measures the strength and direction of a **monotonic relationship** between a feature and a target variable. Unlike Pearson Correlation, it operates on ranked data and can identify relationships that consistently increase or decrease, even when the relationship is nonlinear.

![Spearman_Correlation](/assets/images/Feature_Ranking/spearman.jpg)

> **Relevance to Automotive Feature Ranking**

- Identifies influential variables exhibiting monotonic but nonlinear relationships with the target.
- Well suited for battery, thermal, and powertrain systems where behavior is often nonlinear.
- More robust to noisy sensor measurements and outliers commonly found in test data.
- Captures important dependencies that may be missed by Pearson Correlation.
- Helps prioritize high-impact signals for calibration, validation, and performance optimization.
- Serves as an effective intermediate

## **#3 ReliefF Importance**
ReliefF is a feature-ranking algorithm that evaluates how well a feature distinguishes between similar observations with different target values. Unlike correlation-based methods, ReliefF considers the local neighborhood of data points and can detect complex relationships between features and the target.

![RF_Correlation](/assets/images/Feature_Ranking/reliefF.jpg)

> **Relevance to Automotive Feature Ranking**

- Captures nonlinear relationships commonly observed in vehicle systems.
- Effective for identifying variables influencing battery, thermal, and powertrain performance.
- Can detect interacting signals that jointly affect a target parameter.
- Useful when physical behavior depends on operating conditions and control strategies.
- Helps uncover important variables missed by Pearson or Spearman analysis.
- Well suited for narrowing thousands of logged signals to a smaller set of high-impact features.

## **#4 mRMR Importance**
mRMR (Minimum Redundancy Maximum Relevance) is a feature-selection technique that aims to identify variables that are highly relevant to the target while minimizing redundancy among selected features.
Unlike correlation-based ranking methods that evaluate each feature independently, mRMR considers both feature-target relevance and feature-feature dependency. The objective is to select a compact set of informative and non-duplicative features.

![RF_Correlation](/assets/images/Feature_Ranking/mRMR.jpg)

> **Relevance to Automotive Feature Ranking**

- Particularly useful when hundreds of logged signals contain overlapping information.
- Eliminates redundant variables that represent the same physical behavior.
- Helps construct compact and informative feature sets for calibration activities.
- Reduces engineering effort by focusing on unique contributors to target performance.
- Effective for battery, thermal, electrical, and powertrain investigations.
- Enables selection of a high-quality subset of 50–100 features from thousands of recorded signals.


## **#5 Random Forest Importance**
Random Forest Importance is a machine-learning-based feature-ranking method that evaluates the contribution of each feature toward predicting a target variable. It is derived from an ensemble of decision trees and can capture nonlinear relationships and feature interactions.

![RF_Correlation](/assets/images/Feature_Ranking/randomForest.jpg)

> **Relevance to Automotive Feature Ranking**

- Well suited for complex vehicle datasets containing thousands of logged signals.
- Captures nonlinear effects present in battery, thermal, and electrical systems.
- Identifies interacting variables that influence performance targets.
- Provides reliable rankings even when signal relationships are complex.
- Frequently used for calibration support and root-cause investigations.
- Effective for reducing large signal sets to the most influential variables for detailed engineering analysis.


## **#6 Boruta Importance**
Boruta is an all-relevant feature selection method built around Random Forest. Unlike methods that identify only the minimum set of features required for prediction, Boruta aims to identify **all features that have a statistically significant influence** on the target variable.

The algorithm compares the importance of real features against randomized copies, called **shadow features**, and retains only those features that consistently outperform the random baseline.
![Boruta_Correlation](/assets/images/Feature_Ranking/boruta.jpg)

> **Relevance to Automotive Feature Ranking**

- Helps identify all variables that may influence a vehicle performance target.
- Particularly useful when missing an important signal could impact calibration quality.
- Captures nonlinear behaviors common in battery, thermal, and electrical systems.
- Handles complex interactions between control, sensor, and actuator signals.
- Provides statistically validated feature selection for engineering investigations.
- Well suited for reducing thousands of logged signals to a reliable set of high-impact variables.
---

## **#7 Permutation Importance**
Permutation Importance is a model-based feature-ranking technique that measures how much a model's prediction performance deteriorates when a feature is randomly shuffled.
The underlying idea is simple: if shuffling a feature significantly reduces model accuracy, then that feature must contain important information about the target variable.
Unlike Random Forest Importance, Permutation Importance evaluates a feature's contribution directly through its impact on model performance.
![Boruta_Correlation](/assets/images/Feature_Ranking/perImp.jpg)

> **Relevance to Automotive Feature Ranking**

- Measures the actual contribution of a signal toward prediction accuracy.
- Effective for ranking variables influencing thermal, electrical, and performance targets.
- Captures nonlinear effects commonly observed in vehicle systems.
- Helps validate whether highly ranked features genuinely impact model predictions.
- Useful for calibration, root-cause analysis, and performance optimization studies.
- Enables robust prioritization of the most influential variables from large vehicle-testing datasets.


Every feature selection method represents a different philosophy:

| Method | Key Question |
|----------|-------------|
| Pearson Correlation | Is the relationship linear? |
| Spearman Correlation | Is the relationship monotonic? |
| ReliefF | Does the feature separate local behaviors? |
| Random Forest Importance | Does it improve decisions? |
| mRMR | Does it provide new information? |
| Boruta | Is it better than noise? |
| Permutation Importance | Does the model fail without it? |

No single method provides the complete picture.
Each technique evaluates feature importance from a different perspective and therefore captures different aspects of the underlying system behavior. The strongest feature selection pipelines combine multiple methods into an ensemble ranking framework, leveraging the strengths of each technique while mitigating individual weaknesses. Combining multiple methods enables a more robust and reliable identification of influential variables, particularly in complex automotive systems where linear, nonlinear, interaction-based, and redundant relationships often coexist.


# Ensemble Score Estimation

## Why is an Ensemble Score Needed?

Each feature-ranking method evaluates feature importance from a different perspective. Pearson and Spearman focus on statistical relationships, ReliefF identifies local patterns, Random Forest captures nonlinear interactions, mRMR reduces redundancy, Boruta validates relevance against noise, and Permutation Importance measures impact on model performance.

As a result, the ranking of a feature can vary significantly across methods. Relying on a single technique may overlook important variables or introduce bias toward a specific type of relationship.

An **ensemble score** combines the strengths of multiple feature-ranking methods into a single, more robust metric. This approach improves feature-selection reliability, reduces method-specific bias, and increases confidence that highly ranked features genuinely influence the target variable.

For large automotive datasets containing

To improve feature-ranking reliability, normalized scores from multiple methods (Pearson, Spearman, ReliefF, Random Forest, mRMR, Boruta, and Permutation Importance) are combined into a single ensemble score.

## Average Ensemble Score

The primary ranking score is computed as:

$$
E_{avg} = \frac{1}{n}\sum_{i=1}^{n}s_i
$$

where:

- $$s_i$$ = Normalized score from feature-ranking method $$i$$
- $$n$$ = Number of feature-ranking methods

**Interpretation:** Measures the overall evidence supporting a feature across all methods.

---

## Geometric Mean Score

A confidence score is computed as:

$$
E_{geo} = \left(\prod_{i=1}^{n}s_i\right)^{1/n}
$$

**Interpretation:** Measures the level of agreement among the different feature-ranking methods.

---

## Recommended Interpretation

| Ensemble Score | Meaning |
|---------------|---------|
| High $$E_{avg}$$ + High $$E_{geo}$$ | Strong feature with broad consensus |
| High $$E_{avg}$$ + Low $$E_{geo}$$ | Important feature, but methods disagree |
| Low $$E_{avg}$$ + Low $$E_{geo}$$ | Weak feature with limited evidence |

For automotive feature ranking, $$E_{avg}$$ can be used as the **primary ranking metric**, while $$E_{geo}$$ serves as a **confidence indicator**, providing both feature importance and consensus across methods.
