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
