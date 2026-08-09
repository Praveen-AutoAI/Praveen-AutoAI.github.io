---
layout: post
title: "Discovering Lithium-Ion Cell Calendar-Ageing Rates Using Inverse Physics-Informed Neural Networks (iPINNs)"
date: 2026-08-04
categories: [Machine Learning, Engineering]
tags: [PINN, Physics, Deep Learning, AI]
---

  
## Introduction

### What are Inverse PINNs (iPINNs)?

> A class of **Scientific Machine Learning (SciML)** that embeds governing equations (ODEs/PDEs) into the loss function to infer unknown parameters from data.

- **Forward PINNs:** Given parameters and equations → solve for **state variables**.
- **Inverse PINNs:** Given physical laws and sparse/noisy data → estimate **unknown parameters and degradation rates**.

---

Inverse Physics-Informed Neural Networks (inverse PINNs) refer to a class of **Scientific Machine Learning (SciML)** for inferring unknown parameters, fields, or source terms in differential equation-governed systems by embedding physical knowledge into the loss function of neural network surrogates.

Unlike forward PINNs, which solve for state variables given equations and parameters, inverse PINNs leverage physical laws and sparse or noisy observations to estimate unknown constitutive parameters, source terms, latent fields, or structural design variables.

This class of methods is foundational for scientific machine learning, enabling data-efficient and physically consistent solutions to inverse problems in systems modeled by ordinary and partial differential equations.

### Key Capabilities and Usage

- **Parameter Recovery & Inference** *(e.g., calendar ageing kinetic parameters)*
- **System Identification** *(discovering governing terms/fields)*
- **Robustness with Sparse and Noisy Data**
- **Uncertainty Quantification**

## Problem Statement

> Battery calendar ageing is strongly influenced by storage temperature. Engineers often require degradation-rate parameters to:
>
> - Predict and forecast battery lifetime
> - Understand underlying ageing mechanisms
> - Compare battery performance across different operating conditions

Traditional machine learning approaches typically require large amounts of data and are often criticized as **black-box models** with limited physical interpretability. Conversely, classical curve-fitting methods can reproduce experimental observations but do not explicitly enforce the underlying physics governing the degradation process.

The **objective** of this work is to develop an **Inverse Physics-Informed Neural Network (iPINN)** framework capable of accurately recovering battery degradation parameters directly from sparse experimental observations while maintaining consistency with established battery-ageing theories.

Lithium-ion cell ageing can generally be categorized into two dimensions:

1. **Calendar Ageing**: Degradation occurring during storage, even when the battery is not cycled.
2. **Cyclic Ageing**: Degradation resulting from charge-discharge cycling.

Although battery degradation is a complex, multidimensional process that is not yet fully understood, several successful mathematical hypotheses have been proposed to describe key degradation mechanisms, including:

- Solid Electrolyte Interphase (**SEI**) growth
- Lithium plating
- Electrode cracking
- Electrolyte oxidation

To demonstrate the concept, potential, and elegance of **PINNs** and **inverse PINNs**, this study focuses on a **calendar-ageing dataset**, where battery capacity loss is analyzed as a function of storage temperature.

The goal is to infer physically meaningful degradation-rate parameters from limited observations while leveraging known governing equations to constrain the learning process.
