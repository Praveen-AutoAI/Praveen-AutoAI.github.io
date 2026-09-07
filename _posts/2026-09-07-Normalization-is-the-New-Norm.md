---
layout: post
title: "New Norm is to Normalize - Types and Methods of Normalization in ML Models"
description: "Why we normalize data? How it affects the training process"
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning,]
tags: [Data Science, Deep Learning, AI]
math: true
---


### Normalization of Data

Normalization is the process of transforming data, weights, or activations to a standardized scale so that their magnitudes remain within a controlled range.

Mathematically, normalization often involves centering and scaling a variable:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

where:
* $\mu$ = mean
* $\sigma$ = standard deviation

<p style="color:blue;">
<strong>Remember This:</strong> The goal is not to change the information contained in the data, but to make its numerical representation more suitable for computation/optimization.
</p>

In deep learning, normalization can be applied to:

- Input data **(feature normalization)**
- Intermediate activations **(BatchNorm, LayerNorm, RMSNorm)**
- Model parameters/weights **(WeightNorm, SpectralNorm)**

- 
