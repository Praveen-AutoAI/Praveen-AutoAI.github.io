---
layout: post
title: "Test"
date: 2026-08-09
categories: [Machine Learning, Engineering]
tags: [PINN, Physics, Deep Learning, AI]
math: true
---

## Introduction

### What are Inverse PINNs (iPINNs)?

> A class of **Scientific Machine Learning (SciML)** that embeds governing equations (ODEs/PDEs) into the loss function to infer unknown parameters from data.

- **Forward PINNs:** Given parameters and equations $\rightarrow$ solve for **state variables**.
- **Inverse PINNs:** Given physical laws and sparse/noisy data $\rightarrow$ estimate **unknown parameters and degradation rates**.

---

Inverse Physics-Informed Neural Networks (inverse PINNs) refer to a class of **Scientific Machine Learning (SciML)** methods used for inferring unknown parameters, fields, or source terms in differential equation-governed systems by embedding physical knowledge into the loss function of neural network surrogates.

Unlike forward PINNs, which solve for state variables given equations and parameters, inverse PINNs leverage physical laws and sparse or noisy observations to estimate unknown constitutive parameters, source terms, latent fields, or structural design variables.

This class of methods is foundational for scientific machine learning, enabling data-efficient and physically consistent solutions to inverse problems in systems modeled by ordinary and partial differential equations.

### Key Capabilities and Usage

* **Parameter Recovery & Inference** *(e.g., calendar ageing kinetic parameters)*
* **System Identification** *(discovering governing terms/fields)*
* **Robustness with Sparse and Noisy Data**
* **Uncertainty Quantification**

---

## Problem Statement

> Battery calendar ageing is strongly influenced by storage temperature. Engineers often require degradation-rate parameters to:
>
> - Predict and forecast battery lifetime
> - Understand underlying ageing mechanisms
> - Compare battery performance across different operating conditions

Traditional machine learning approaches typically require large amounts of data and are often criticized as **black-box models** with limited physical interpretability. Conversely, classical curve-fitting methods can reproduce experimental observations but do not explicitly enforce the underlying physics governing the degradation process.

The **objective** of this work is to develop an **Inverse Physics-Informed Neural Network (iPINN)** framework capable of accurately recovering battery degradation parameters directly from sparse experimental observations while maintaining consistency with established battery-ageing theories.

Lithium-ion cell ageing can generally be categorized into two dimensions:

1. **Calendar Ageing:** Degradation occurring during storage, even when the battery is not cycled.
2. **Cyclic Ageing:** Degradation resulting from charge-discharge cycling.

Although battery degradation is a complex, multidimensional process that is not yet fully understood, several successful mathematical hypotheses have been proposed to describe key degradation mechanisms, including:

* Solid Electrolyte Interphase (**SEI**) growth
* Lithium plating
* Electrode cracking
* Electrolyte oxidation

To demonstrate the concept, potential, and elegance of **PINNs** and **inverse PINNs**, this study focuses on a **calendar-ageing dataset**, where battery capacity loss is analyzed as a function of storage temperature.

The goal is to infer physically meaningful degradation-rate parameters from limited observations while leveraging known governing equations to constrain the learning process.

---

## Dataset

The experimental data used in this **Inverse Physics-Informed Neural Network (iPINN)** demonstration corresponds to calendar-ageing measurements of a commercial **Lithium Iron Phosphate (LFP)** cell. The objective of the original study was to investigate the impact of **storage temperature** on battery degradation under high State-of-Charge (SoC) conditions.

### Cell Specifications

The experiments were performed on a commercial **Sony US26650FTC1** cell designed for stationary energy-storage applications.

| Characteristic | Specification |
| :--- | :--- |
| **Format** | 26650 Cylindrical Cell |
| **Chemistry** | Lithium Iron Phosphate ($LiFePO_4$) |
| **Nominal Capacity** | 3000 mAh |
| **Nominal Voltage** | 3.2 V |
| **Operating Voltage Range** | 2.0 V to 3.6 V |

---

### Calendar Ageing Test Conditions

* **Storage State:** Constant **100% State of Charge (SoC)**
* **Temperature Matrix:** **15°C, 25°C, 35°C, and 45°C**
* **Time Span:** Day 0 to approximately Day 235 (sparse observation checkpoints)

The original study reports **capacity loss** over time. For the iPINN implementation, the data was converted to **remaining capacity**, which serves as the target variable for the neural network:

$$Q = 1 - Loss$$

where:
* $Q$ = Remaining normalized capacity
* $Loss$ = Measured capacity fade

Using remaining capacity instead of capacity loss allows the network to directly model battery **State-of-Health (SoH)** evolution while simultaneously enforcing the underlying degradation physics through the PINN framework.

### Experimental Data: Impact of Temperature on Calendar Ageing

![Experimental Data Impact of Temperature on Calendar Ageing](/assets/images/Experimental_Data.jpg)

---

### Source

The experimental data was extracted from:

> Naumann, M. et al., *Analysis and Modeling of Calendar Aging of a Commercial $LiFePO_4$/Graphite Cell*, Journal of The Electrochemical Society.
> 
> **Paper Link:** [IOP Science](https://iopscience.iop.org/article/10.1149/2.1181714jes)

---

## Physics Model

The degradation of lithium-ion batteries is a complex, multidimensional phenomenon involving several interacting ageing mechanisms. While numerous degradation pathways exist, calendar ageing is primarily driven by **SEI (Solid Electrolyte Interphase) growth** and other parasitic side reactions occurring at the electrode-electrolyte interface.

The objective of this work is not to develop a comprehensive battery degradation model, but rather to demonstrate how an **Inverse Physics-Informed Neural Network (iPINN)** can identify physically meaningful degradation parameters directly from sparse experimental observations.

To keep the physics simple and interpretable, we assume that battery capacity loss arises from two primary mechanisms:
* Diffusion-limited SEI growth
* Long-term parasitic ageing reactions

---

### Step 1: Capacity-Loss Model

A commonly used semi-empirical representation of calendar ageing is expressed as:

$$Loss(t) = a\sqrt{t} + bt$$

where:
* $a$ = degradation parameter associated with diffusion-limited SEI growth
* $b$ = degradation parameter associated with effective long-term linear-ageing contributions
* $t$ = storage time

The $\sqrt{t}$ term captures rapid initial degradation associated with diffusion-limited SEI growth, while the linear term ($bt$) represents slower, continuous long-term ageing mechanisms.
