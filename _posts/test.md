---
layout: post
title: "Physics-Informed Neural Networks: How Forward and Inverse PINNs Solve Small-Data Engineering Problems"
description: "A beginner-friendly introduction to Forward and Inverse Physics-Informed Neural Networks (PINNs) with engineering applications."
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning]
tags: [Data Science, Deep Learning, AI, PINN]
math: true
---

# Paths to PINNs: Forward and Inverse Approaches for Solving Engineering Problems with Small Datasets

## 1. Motivation and Real-World Relevance

Consider a metal rod used in an industrial heating system. The rod is heated at one end, and engineers need to understand how heat propagates through the material.

Temperature sensors can be installed at only a few accessible locations. Installing sensors throughout the entire rod is often impractical, expensive, or impossible. In addition, the effective thermal conductivity of the material may not be accurately known due to manufacturing variability, ageing, or changing operating conditions.

As a result, the available experimental information is incomplete.

Engineers may only have:

- A small number of temperature measurements
- Knowledge of the heating conditions
- The governing heat-transfer equation

From this limited information, they would like to determine:

1. The complete temperature distribution throughout the rod
2. The unknown thermal conductivity of the material

This challenge appears across many engineering domains:

- Battery engineers estimate internal cell temperatures and heat-generation rates using only surface measurements.
- Aerospace engineers infer structural degradation from sparse strain measurements.
- Manufacturing engineers estimate unknown heat-source characteristics from thermal imaging.
- Energy engineers reconstruct underground permeability from measurements collected at a limited number of wells.

### Common Characteristics of These Problems

These applications share four key characteristics:

- **Sparse measurements**: Only a limited number of sensors are available.
- **Known physics**: Governing equations describing system behavior are available.
- **Unknown quantities**: Parameters or internal states cannot be measured directly.
- **Reconstruction objective**: Infer the unknown quantities from data and physics.

## Overview: How Inverse PINNs Solve This Problem

<div style="text-align:center;">
/assets/images/PINN/I_PINN_1.png
</div>

---

# 2. Forward vs. Inverse Problems

To understand Physics-Informed Neural Networks, it is important to distinguish between **forward problems** and **inverse problems**.

These represent two fundamentally different ways of solving engineering problems.

## The Forward Problem

The forward problem represents the traditional engineering simulation workflow.

In a forward problem:

- Material properties are known.
- Boundary conditions are known.
- Governing equations are known.

The objective is to compute the resulting system response.

Examples include:

- Computing temperature distribution from known thermal conductivity.
- Computing structural deformation from known loads and material properties.
- Computing fluid velocity and pressure from known flow conditions.

Traditional simulation tools such as:

- Finite Element Analysis (FEA)
- Computational Fluid Dynamics (CFD)
- Finite Difference Methods (FDM)

are designed to solve forward problems.

```text
Known Physics + Known Parameters
                ↓
            Solve PDE
                ↓
      System State / Response
```

---

## The Inverse Problem

The inverse problem reverses the workflow.

Some physical quantities are unknown, but limited measurements of the system response are available.

The objective is to infer the hidden quantities that produced the observed measurements.

Examples include:

- Estimating thermal conductivity from temperature measurements
- Estimating material degradation from strain measurements
- Identifying unknown heat sources from thermal images
- Estimating underground permeability from pressure measurements

```text
Sparse Measurements + Known Physics
                  ↓
     Parameter Identification
                  ↓
   Unknown Physical Quantities
```

### U-Turn: Forward vs. Inverse PINNs

<div style="text-align:center;">
/assets/images/PINN/I_PINN_4.png
</div>

Unlike traditional curve fitting, inverse problems must remain consistent with the governing physics.

This makes inverse problems more challenging but also far more valuable for:

- Digital twins
- Health monitoring
- Condition estimation
- Scientific discovery

---

# 3. Core Differences Between Forward and Inverse PINNs

| Feature | Forward PINN | Inverse PINN |
|----------|----------|----------|
| Objective | Predict state fields | Estimate unknown parameters and state fields |
| Physical Parameters | Known | Unknown and trainable |
| Trainable Variables | Neural network weights | Neural network weights + physical parameters |
| Data Requirement | ICs and BCs | Sparse measurements + ICs/BCs |
| Main Output | System response | Parameters + system response |
| Classical Alternative | FEA / CFD | Optimization-based parameter identification |

## In Simple Terms

> A Forward PINN uses known physics and known parameters to predict the system state.

> An Inverse PINN uses sparse observations and governing physics to estimate unknown parameters while simultaneously reconstructing the system state.

---

# 4. Common Foundations Shared by Forward and Inverse PINNs

Although their objectives differ, both approaches share the same mathematical framework.

| Common Component | Description |
|------------------|-------------|
| Neural Networks | Represent unknown solution fields |
| Automatic Differentiation | Computes PDE derivatives exactly |
| Physics Loss | Enforces governing equations |
| Collocation Points | Apply physics throughout the domain |
| Gradient-Based Optimization | Adam, L-BFGS, etc. |
| Physics Regularization | Reduces overfitting and improves generalization |

## Key Takeaway

> Forward PINNs and Inverse PINNs use the same learning framework.

> The primary difference is **what is being learned**.

- Forward PINNs learn the state field.
- Inverse PINNs learn both the state field and unknown physical parameters.

---

# 5. PINN Loss Function Design

The key innovation behind a Physics-Informed Neural Network lies in its loss function.

> A neural network should not only fit the data, but also obey the laws of physics.

Traditional neural networks learn only from observations.

PINNs learn from observations and governing equations simultaneously.

---

## Traditional Neural Network

A conventional neural network minimizes only data error:

$$
L = L_{data}
$$

For example:

$$
L_{data}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_{pred,i}-y_{true,i})^2
$$

The objective is simple:

> Fit the available data.

---

## Introducing Physics

Suppose the system is governed by:

$$
\mathcal{N}(u)=0
$$

where:

- $u$ is the unknown solution
- $\mathcal{N}$ is the governing differential operator

The neural network predicts:

$$
u_\theta(x,t)
$$

A PDE residual is then defined as:

$$
R = \mathcal{N}(u_\theta)
$$

When physics is perfectly satisfied:

$$
R = 0
$$

---

## Physics Loss

Any violation of the governing PDE is penalized:

$$
L_{physics}
=
\frac{1}{N}
\sum_{i=1}^{N}
R_i^2
$$

This tells the neural network:

> Do not merely fit the measurements. Also satisfy the governing physics.

---

## Basic PINN Loss

The first PINN loss function becomes:

$$
L
=
L_{data}
+
\lambda L_{physics}
$$

where:

- $L_{data}$ measures prediction error
- $L_{physics}$ measures PDE violation
- $\lambda$ balances the two objectives

---

## Adding Boundary Conditions

Suppose the solution must satisfy:

$$
u(0,t)=100
$$

The boundary-condition loss becomes:

$$
L_{BC}
=
\frac{1}{N}
\sum_{i=1}^{N}
(u_{pred,i}-u_{BC,i})^2
$$

The total loss becomes:

$$
L
=
L_{data}
+
L_{physics}
+
L_{BC}
$$

---

## Adding Initial Conditions

For transient systems:

$$
u(x,0)=u_0(x)
$$

The initial-condition loss is:

$$
L_{IC}
=
\frac{1}{N}
\sum_{i=1}^{N}
(u_{pred,i}-u_{0,i})^2
$$

The complete PINN loss becomes:

$$
L
=
L_{data}
+
L_{physics}
+
L_{BC}
+
L_{IC}
$$

---

## Evolution of the PINN Loss Function

### Traditional Neural Network

$$
L=L_{data}
$$

**Goal:** Fit the data

### Basic PINN

$$
L=L_{data}+L_{physics}
$$

**Goal:** Fit the data and satisfy the PDE

### Practical Engineering PINN

$$
L=L_{data}+L_{physics}+L_{BC}+L_{IC}
$$

**Goal:** Fit the data, satisfy the PDE, respect boundary conditions, and satisfy initial conditions.

---

## The Big Picture

A PINN learns from two teachers.

### Teacher 1: Data

$$
L_{data}
$$

Measurements tell the network what happened.

### Teacher 2: Physics

$$
L_{physics}
$$

The governing equations tell the network what is physically possible.

---

## Key Takeaway

$$
\boxed{
L=
L_{data}
+
L_{physics}
+
L_{BC}
+
L_{IC}
}
$$

Unlike a traditional neural network, a PINN learns simultaneously from:

- Experimental data
- Governing PDEs
- Boundary conditions
- Initial conditions

This simple modification enables PINNs to solve many engineering problems using only small amounts of data while remaining physically consistent.
