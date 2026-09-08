---
layout: post
title: "The New Norm is to Normalize - Episode_#3 : Activation Normalization"
description: "Why we normalize data? How it affects the training process"
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning,]
tags: [Data Science, Deep Learning, AI]
math: true
---

## Introduction to Normalization

### What's Normalization?
Normalization is the process of transforming data, weights, or activations to a standardized scale so that their magnitudes remain within a controlled range.

Mathematically, normalization often involves centering and scaling a variable:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

where:
* $\mu$ = mean
* $\sigma$ = standard deviation

In deep learning, normalization can be applied to:

- Input data - **Feature normalization** 
- Model parameters/weights - **Weight Normalization**  
- Output of activations - **Activation Normalization**


<p style="color:blue;">
<strong>Remember This:</strong> The goal is not to change the information contained in the data, but to make its numerical representation more suitable for computation/optimization. Normalization controls the scale of signals inside a neural network, making optimization faster, more stable, and more reliable. Depending on what is being normalized, normalization techniques can be broadly classified into Weight Normalization (normalizing model parameters) and Activation Normalization (normalizing intermediate feature activations).
</p>

### What is the Motivation in the Context of ML Model Training?

Machine learning models learn by processing input data through multiple layers and continuously updating their parameters based on the prediction error. During this process, numerical instabilities can arise from both the input features and the internal computations of the network.

Common challenges include:

- Input features may have vastly different scales (e.g., age in years vs. salary in millions).
- Activations may become excessively large or small as they propagate through layers.
- Gradients can vanish or explode during backpropagation.
- Training becomes highly sensitive to parameter initialization.
- Small parameter updates in one layer can have amplified effects in deeper layers.

As networks become deeper, even slight changes in feature distributions, activation distributions, or weight magnitudes can compound across layers, making optimization increasingly difficult.

Normalization addresses these challenges at different stages of the learning pipeline.
From an optimization perspective, normalization allows gradient descent to focus on learning meaningful patterns rather than constantly adapting to changing signal magnitudes.

### Normalization at Different Stages of the Pipeline
![Normalization_Effect](/assets/images/Normalization/Normalization_4.png)

### Why Normalization Matters

| Benefit | Impact on Training |
| :--- | :--- |
| **Mitigates Vanishing & Exploding Gradients** | Keeps activations and weights bounded so gradients remain numerically stable during backpropagation. |
| **Accelerates Convergence** | Reduces the number of training iterations needed to achieve high accuracy. |
| **Allows Larger Learning Rates** | Makes weight updates more predictable, enabling faster optimization without divergence. |
| **Reduces Initialization Sensitivity** | Makes training less dependent on meticulously chosen initial weight schemes. |
| **Improves Training Stability** | Maintains consistent activation distributions across layers for a smoother loss landscape. |
| **Enables Deep Architectures** | Critical for scaling massive models (e.g., ResNet, Transformer variants like GPT, Llama, Mistral). |


<p style="color:blue;">
<strong>Remember This:</strong> This is the key insight: WeightNorm and SpectralNorm are not trying to change the weight distribution. They control different geometric properties of the weight matrix.
The histogram can look nearly identical while the network behavior changes dramatically. Similar weight distributions do not imply similar network dynamics.
</p>



**Check out my articles** [Episode 1: Feature Normalization Methods][epi1-norm] and [Episode 2: Weight Normalization Methods][epi2-norm] before proceeding with Activation Normalization.

[epi1-norm]: https://praveen-autoai.github.io/machine%20learning/engineering/scientific%20machine%20learning/2026/09/07/Normalization-is-the-New-Norm-Epi1.html
[epi2-norm]: https://praveen-autoai.github.io/machine%20learning/engineering/scientific%20machine%20learning/2026/09/07/Normalization-is-the-New-Norm-Epi2.html

## 3.Activation Normalization 

Activation normalization methods stabilize the **intermediate feature representations (activations outputs of hidden layers)** inside a neural network. By maintaining a consistent scale of activations across layers, they improve gradient flow, accelerate convergence, and enable the stability during training of deeper architectures. Smoothing the loss landscape is an effect of normalization methods

### A. BatchNorm
**Asks : "How does a sample compare to other samples in the batch?"**

BatchNorm normalizes activations using the statistics of the mini-batch. It was introduced to reduce activation distribution drift(internal covariate shift)  during training and improve optimization stability by resetting distributions to zero mean and unit variance before applying a learnable scale and shift. 

#### Core Mechanism & Formulation

Given a mini-batch **B = {x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>m</sub>}** of size *m* across a specific feature dimension:

1. **Calculate Mini-Batch Mean:**
$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

2. **Calculate Mini-Batch Variance:**
$$
\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

3. **Normalize Activations:**
$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$
*(where **ε > 0** is a numerical stability constant preventing division by zero)*

4. **Apply Learnable Scale & Shift (Affine Transformation):**
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   *(where **γ** and **β** are learnable parameters that allow the network to recover original scales if optimal)*

#### Key Properties

* **Training Phase:** Mini-batch statistics (**μ<sub>B</sub>**, **σ<sub>B</sub><sup>2</sup>**) are computed dynamically per batch while updating running exponential averages.
* **Inference Phase:** Mini-batch statistics are bypassed. Fixed global running statistics (**μ<sub>run</sub>**, **σ<sub>run</sub><sup>2</sup>**) ensure deterministic predictions for individual inputs.
* **Optimization Benefit:** Smooths the loss landscape, enables higher learning rates, and mitigates sensitivity to parameter initialization.

--- 

### B. LayerNorm

**Asks: "How does a feature in a sample compare to other features within the same sample?"**

LayerNorm normalizes activations across all feature dimensions for a single data point rather than across the mini-batch. Introduced by Ba et al. to address BatchNorm’s dependencies on batch size, LayerNorm stabilizes hidden representations independently of batch dynamics, making it exceptionally well-suited for sequence models, Transformers, and PINNs.

---

#### Core Mechanism & Formulation

Given a single sample vector **x = [x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>d</sub>]<sup>T</sup>** across *d* feature/hidden dimensions:

1. **Calculate Feature Mean:**
   $$
   \mu_L = \frac{1}{d} \sum_{i=1}^{d} x_i
   $$

2. **Calculate Feature Variance:**
   $$
   \sigma_L^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu_L)^2
   $$

3. **Normalize Activations:**
   $$
   \hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}
   $$
   *(where **ε > 0** is a numerical stability constant preventing division by zero)*

4. **Apply Learnable Scale & Shift (Affine Transformation):**
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   *(where **γ** and **β** are learnable element-wise parameters that allow the network to recover original scales if optimal)*


> #### Key Properties

* **Batch Independence:** Computes statistics strictly per individual sample, enabling identical execution regardless of mini-batch size (including batch size = 1) and sequence length.
* **Training–Inference Consistency:** The exact same mathematical operation is applied during both training and evaluation—eliminating the need to track global running averages (**μ<sub>run</sub>**, **σ<sub>run</sub><sup>2</sup>**).
* **PINN / Scientific ML Advantage:** Avoids coupling distinct spatial/temporal collocation points, ensuring clean autograd computation of exact PDE derivatives (**∂u / ∂x**) without introducing mini-batch noise.


### B. RMSNorm
**Asks : "How large is the overall signal in this sample?"**

<p style="color:blue;">
<strong>Remember This:</strong> 
- BatchNorm asks: "How does this sample compare to other samples in the batch?" - compares samples to other samples
- LayerNorm asks: "How do the features within this sample compare to one another?" - compares features within a sample
- RMSNorm asks: "How large is the overall signal in this sample?" - measures and controls only the overall magnitude of the sample's feature vector
</p>



### Comparison: Activation Normalization Techniques

| Aspect | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) | RMS Normalization (RMSNorm) |
| :--- | :--- | :--- | :--- |
| **Intuition** | Normalize activations using statistics computed from the entire mini-batch. Each sample benefits from information provided by other samples in the batch. | Normalize all features within a single sample, making each sample self-contained and independent of others. | Normalize only the overall signal magnitude (RMS) of a sample without subtracting the mean. |
| **Uniqueness & Advantage** | • Uses batch-level statistics<br>• Acts as both normalization and regularization<br>• Enables higher learning rates<br>• Highly effective for CNNs | • Independent of batch size<br>• No running statistics required<br>• Same behavior during training and inference<br>• Ideal for sequence models and transformers | • Removes mean-centering step from LayerNorm<br>• Fewer computations<br>• Lower memory overhead<br>• Similar performance to LayerNorm with better efficiency |
| **Limitations** | • Performance degrades with small batches<br>• Requires running mean and variance<br>• Synchronizing statistics in distributed training creates overhead<br>• Less suitable for sequence models | • Provides less benefit for CNNs compared to BatchNorm<br>• Slightly more computationally expensive than RMSNorm | • Does not explicitly center activations around zero<br>• Less theoretically studied than LayerNorm<br>• May not be ideal when mean-centering is critical |
| **Where to Use** | • CNNs<br>• Computer Vision models<br>• Large-batch training<br>• Feed-forward networks | • Transformers (BERT, GPT)<br>• RNNs and LSTMs<br>• Variable-length sequences<br>• Small-batch or distributed training | • Modern LLMs (LLaMA, PaLM)<br>• Large-scale transformers<br>• Compute-efficient architectures |
| **Normalization Scope** | Across the Batch Dimension | Across Features within a Sample | Across the RMS of Features |
| **Batch-Size Dependency** | ✅ Depends on Batch Size | ❌ Independent | ❌ Independent |
| **Training vs. Inference** | Different behavior (running statistics used during inference) | Same behavior | Same behavior |

#### Core Mechanism & Formulation

Given a mini-batch **B = {x₁, x₂, ..., xₘ}** of size *m* across a specific feature dimension:

1. **Calculate Mini-Batch Mean:**
   $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

2. **Calculate Mini-Batch Variance:**
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

3. **Normalize Activations:**
   $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
   *(where **ε > 0** is a small constant for numerical stability (e^8))*

4. **Apply Learnable Scale & Shift:**
   $$y_i = \gamma \hat{x}_i + \beta$$
 *(where **γ** [scale] and **β** [shift] allow the network to recover optimal representations)*
