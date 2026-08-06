---
layout: post
title: "Discovering Lithium-Ion Cell Calendar-Ageing Rates Using Inverse Physics-Informed Neural Networks (iPINNs)"
date: 2026-08-04
categories: [Machine Learning, Engineering]
tags: [PINN, Physics, Deep Learning, AI]
---

<h3>Introduction</h3>

<div style="font-size: 16px;
border-left: 5px solid #2e86de;
padding: 12px 18px;
background-color: #f8f9fa;
border-radius: 4px;
">

    
#### What are Inverse PINNs (iPINNs)?
A class of **Scientific Machine Learning (SciML)** that embeds governing equations (ODEs/PDEs) into the loss function to infer unknown parameters from data.

* **Forward PINNs:** Given parameters & equations $\rightarrow$ solve for **state variables**.
* **Inverse PINNs:** Given physical laws & sparse/noisy data $\rightarrow$ estimate **unknown parameters & degradation rates**.

---

Inverse Physics-Informed Neural Networks (inverse PINNs) refer to a class of Scientific Machine Learning(SciML) for inferring unknown parameters, fields, or source terms in differential equation-governed systems by embedding physical knowledge into the loss function of neural network surrogates. Unlike forward PINNs, which solve for state variables given equations and parameters, inverse PINNs leverage physical laws and sparse or noisy observations to estimate unknown constitutive parameters, source terms, latent fields, or structural design variables. This class of methods is foundational for scientific machine learning, enabling data-efficient and physically consistent solutions to inverse problems in systems modeled by ordinary and partial differential equations.


#### Key Capabilities/Usage:
- Parameter Recovery & Inference *(e.g., calendar ageing kinetic parameters)*
- System Identification *(discovering governing terms/fields)*
- Robustness with Sparse & Noisy Data
- Uncertainty Quantification


<h3>Problem Statement</h3>

<div style="font-size: 16px;
border-left: 5px solid #2e86de;
padding: 12px 18px;
background-color: #f8f9fa;
border-radius: 4px;
">

Battery calendar ageing is strongly influenced by storage temperature. Engineers often require degradation-rate parameters to:

- Predict and forecast battery lifetime
- Understand underlying ageing mechanisms
- Compare battery performance across different operating conditions

Traditional machine learning approaches typically require large amounts of data and are often criticized as **black-box models** with limited physical interpretability. On the other hand, classical curve-fitting methods can match experimental observations but do not explicitly enforce the underlying physics governing the degradation process.

This project investigates whether an **Inverse Physics-Informed Neural Network (iPINN)** can accurately recover battery degradation parameters directly from sparse experimental observations while maintaining physical consistency with established battery-ageing theory.
