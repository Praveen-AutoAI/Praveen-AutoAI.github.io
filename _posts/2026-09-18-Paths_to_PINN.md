---
layout: post
title: "Physics-Informed Neural Networks : How Forward and Inverse PINN can solve small and very small dataset problem "
description: "Inverse Physics-Informed Neural Networks (Inverse PINNs): A Beginner-Friendly Introduction with Engineering Applications"
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning,]
tags: [Data Science, Deep Learning, AI]
math: true
---

## Paths to PINN (Physics-Informed Neural Networks) : Forward and Inverse ways of Solving Engineering Problems with (very)Small Dataset
### 1. Motivation and Real-World Relevance

Consider a metal rod used as part of an industrial heating system. The rod is heated at one end, and engineers need to understand how quickly heat travels through the material. Temperature sensors can be installed at a few accessible locations, but placing sensors at every point along the rod is neither practical nor necessary. More importantly, the rod's effective thermal conductivity may be unknown because of manufacturing variation, material degradation, or uncertain operating conditions.

The **available information from experiment is therefore incomplete**. Engineers have a small number of temperature measurements, some knowledge of the heating conditions, and a physical law describing heat conduction. From this limited information, they would like to determine two things:

1. The complete temperature distribution throughout the rod.
2. The unknown thermal conductivity of the material.

This type of challenge appears throughout engineering:

- Battery engineers may need to **estimate internal cell temperatures and heat-generation rates using only surface sensors.**
- Aerospace engineers may **infer material degradation from strain measurements.**
- Manufacturing engineers may **estimate heat-source characteristics from thermal camera data**.
- Energy engineers may reconstruct subsurface permeability using measurements from a small number of wells.

## These problems share a common structure:
- Sparse measurements: Limited data from a few locations or sensors.
- Physical laws: Governing equations that describe the system's behavior.
- Unknowns to estimate: Parameters or states that are not directly measurable.
- Reconstruction: Using the available data and physical laws to infer the unknowns.

## Overview: How Inverse-PINNs solve this!
<div style="text-align: center;">
  <img src="/assets/images/PINN/I_PINN_1.png" alt="PINN-1" width="800">
</div>

---

# 2. Forward vs. Inverse Problems

To understand inverse Physics-Informed Neural Networks (inverse PINNs), it is important to distinguish between **forward problems** and **inverse problems**, which represent two fundamentally different modeling paradigms in engineering and scientific computing.

## The Forward Problem

The **forward problem represents the traditional engineering simulation workflow**. In this setting, the geometry, material properties (e.g., thermal conductivity), boundary conditions (e.g., applied heat flux), and governing partial differential equations (PDEs) are known.

The **objective is to compute the resulting state field of the system.**

For example:

- Given a rod's thermal conductivity and heating conditions, compute the temperature distribution.
- Given a structure's material properties and loading conditions, compute the stress and deformation fields.
- Given fluid properties and inlet conditions, compute the velocity and pressure fields.

Tools such as **Finite Element Analysis (FEA)** and **Computational Fluid Dynamics (CFD)** are specifically designed to solve forward problems.

```text
Known Physics + Known Parameters
                ↓
         Solve PDEs
                ↓
       System Response/State Field
```

---

## The Inverse Problem

The inverse problem reverses this workflow. In this setting, some physical properties, boundary conditions, or internal source terms are unknown. However, measurements of the system's response are available at a limited number of locations.

The **objective is to work backward from these observations to infer the unknown quantities that produced them.**

For example:

- Estimate thermal conductivity from a few temperature measurements.
- Infer material degradation from strain sensor data.
- Identify unknown heat-source characteristics from thermal imaging.
- Estimate subsurface permeability from pressure measurements in wells.

```text
Sparse Measurements + Known Physics
                  ↓
     Parameter Identification
                  ↓
   Unknown Physical Quantities
```

## U-Turn : Forward and Inverse PINNs
<div style="text-align: center;">
  <img src="/assets/images/PINN/I_PINN_4.png" alt="PINN Difference" width="800">
</div>

Unlike simple curve fitting, inverse problems must produce solutions that remain consistent with the governing physics of the system. This requirement makes inverse problems considerably more challenging, but also more valuable for engineering analysis, system monitoring, and digital twin applications.

# Core Differences: Forward PINN vs. Inverse PINN

| **Feature / Dimension** | **Forward PINN** | **Inverse PINN (I-PINN)** |
|-------------------------|------------------|---------------------------|
| **Primary Objective** | Solve the governing PDE to predict system state fields $u(x,t)$ across the domain. | Infer unknown physical parameters $\lambda$ while simultaneously reconstructing state fields $u(x,t)$. |
| **Physical Parameters ($\lambda$)** | **Known and fixed** constants in the governing differential equation. | **Unknown and learnable** variables initialized with initial guesses. |
| **Trainable Variables** | Network weights and biases $(W,b)$ only. | Network weights and biases $(W,b)$ **plus** physical parameters $(\lambda)$. |
| **Primary Data Source** | Known initial conditions (ICs) and boundary conditions (BCs). | Sparse, noisy interior sensor measurements (e.g., thermocouples, accelerometers) alongside available ICs/BCs. |
| **Role of Sensor Data** | Anchors domain boundaries and starting state. | Provides observational evidence to calibrate unknown physical constants and fill state gaps. |
| **Traditional Alternative** | Numerical discretization solvers (FEA, CFD, Finite Difference Methods). | Iterative optimization loops wrapping thousands of repeated forward FEA/CFD runs. |
| **Optimization Goal** | Adjust $(W,b)$ until $u(x,t)$ satisfies both boundary data and PDE residuals. | Adjust $(W,b)$ and $\lambda$ simultaneously until predicted states match measurements and satisfy the PDE. |

---

In Simple Terms:
> A **Forward PINN** uses known physical parameters to solve for the system state.
> 
> An **Inverse PINN** uses sparse observations and known governing physics to simultaneously estimate both the system state and the unknown physical parameters.

# Common Features (Shared Foundations)

Although **Forward PINNs** and **Inverse PINNs** are designed for different objectives, they share the same underlying computational framework and physics-informed learning principles.

| **Common Aspect** | **Description across Both Paradigms** |
|-------------------|----------------------------------------|
| **Mesh-Free Representation** | Both use deep neural networks that take continuous spatial coordinates $(x,y,z)$ and time $(t)$ as inputs and output physical state variables $u$. |
| **Automatic Differentiation (AD)** | Both leverage automatic differentiation to compute exact analytical derivatives such as $\frac{\partial u}{\partial t}$ and $\frac{\partial^2 u}{\partial x^2}$ without grid discretization or finite-difference approximations. |
| **Composite Loss Structure** | Both employ a multi-component loss function that combines data fidelity and physics constraints. |
| **Physics Regularization** | Both evaluate the governing PDE residual at randomly sampled collocation points $(x_m,t_m)$ to ensure physically meaningful solutions throughout the domain. |
| **Optimization Algorithms** | Both use gradient-based optimization methods such as Adam and L-BFGS to minimize the total loss and update learnable parameters through backpropagation. |
| **Noise Robustness** | Both exploit governing physical equations as regularizing constraints, helping suppress measurement noise and reduce overfitting. |


## Key Takeaway

> Despite their different objectives, both **Forward PINNs** and **Inverse PINNs** rely on the same foundational components: neural-network function approximation, automatic differentiation, PDE-constrained learning, and gradient-based optimization.

> The primary distinction lies not in the architecture itself, but in **what is being learned**. A **Forward PINN** learns the **state field** $u(x,t)$ when physical parameters are known, whereas an **Inverse PINN** learns both the **state field** and the **unknown physical parameters** simultaneously from sparse observations.

# PINN Loss Function Design for General PDEs

Physics-Informed Neural Networks (PINNs) solve Partial Differential Equations (PDEs) by embedding the governing physics directly into the neural network training process. Unlike conventional neural networks that rely purely on labeled data, PINNs optimize a composite loss function that simultaneously satisfies:

- Observational data
- Governing PDEs
- Initial conditions (ICs)
- Boundary conditions (BCs)

The design of the loss function is the most critical aspect of a PINN because it determines how well the solution respects both the measurements and the underlying physics.

---

## General PDE Formulation

Consider a generic PDE:

$$
\mathcal{N}[u(\mathbf{x},t;\lambda)] = 0
\qquad
(\mathbf{x},t)\in\Omega
$$

where:

- $u(\mathbf{x},t)$ is the unknown state variable
- $\mathbf{x}$ denotes the spatial coordinates
- $t$ denotes time
- $\lambda$ represents physical parameters
- $\mathcal{N}[\cdot]$ is a differential operator
- $\Omega$ is the computational domain

Examples include:

| PDE | State Variable |
|-------|-------|
| Heat Equation | Temperature |
| Wave Equation | Displacement |
| Burgers Equation | Velocity |
| Navier-Stokes Equation | Velocity, Pressure |
| Maxwell Equations | Electric and Magnetic Fields |

---

# Neural Network Approximation

The neural network approximates the solution as:

$$
u_\theta(\mathbf{x},t)
$$

where:

$$
\theta = \{W,b\}
$$

represents all trainable weights and biases.

The objective of training is to find:

$$
\theta^*
=
\arg \min L
$$

such that the learned solution satisfies both measurements and physics.

---

# PDE Residual Loss

Using automatic differentiation, all required derivatives are computed directly from the network.

The PDE residual is defined as

$$
R(\mathbf{x},t)
=
\mathcal{N}
\left[
u_\theta(\mathbf{x},t)
\right]
$$

Ideally:

$$
R(\mathbf{x},t)=0
$$

everywhere in the solution domain.

The Physics Loss becomes:

$$
L_{\text{PDE}}
=
\frac{1}{N_r}
\sum_{i=1}^{N_r}
R(\mathbf{x}_i,t_i)^2
$$

where:

- $N_r$ = number of collocation points
- $(\mathbf{x}_i,t_i)$ = interior sampling points

This term forces the network to satisfy the governing PDE.

---

# Initial Condition Loss

For transient PDEs, the solution at the initial time is known.

Given

$$
u(\mathbf{x},0)
=
u_0(\mathbf{x})
$$

the Initial Condition Loss is

$$
L_{\text{IC}}
=
\frac{1}{N_{IC}}
\sum_{i=1}^{N_{IC}}
\left(
u_\theta(\mathbf{x}_i,0)
-
u_0(\mathbf{x}_i)
\right)^2
$$

This ensures the network starts from the physically correct state.

---

# Boundary Condition Loss

The network must satisfy boundary conditions.

---

## Dirichlet Boundary Condition

Given:

$$
u(\mathbf{x},t)
=
g(\mathbf{x},t)
\qquad
\text{on }
\partial\Omega
$$

the loss becomes

$$
L_{\text{BC}}
=
\frac{1}{N_{BC}}
\sum_{i=1}^{N_{BC}}
\left(
u_\theta
-
g
\right)^2
$$

---

## Neumann Boundary Condition

Given:

$$
\frac{\partial u}{\partial n}
=
q
$$

the loss becomes

$$
L_{\text{BC}}
=
\frac{1}{N_{BC}}
\sum_{i=1}^{N_{BC}}
\left(
\frac{\partial u_\theta}{\partial n}
-
q
\right)^2
$$

---

## Robin Boundary Condition

Given:

$$
a u + b \frac{\partial u}{\partial n}
=
c
$$

the corresponding loss becomes

$$
L_{\text{BC}}
=
\frac{1}{N_{BC}}
\sum_{i=1}^{N_{BC}}
\left(
a u_\theta
+
b\frac{\partial u_\theta}{\partial n}
-
c
\right)^2
$$

---

# Data Loss

If measurements are available, the network can be constrained using data.

Suppose experimental observations are

$$
\{u_{\text{meas}}\}
$$

Then:

$$
L_{\text{data}}
=
\frac{1}{N_d}
\sum_{i=1}^{N_d}
\left(
u_{\text{pred}}
-
u_{\text{meas}}
\right)^2
$$

This term improves prediction accuracy and reduces ambiguity.

---

# Composite PINN Loss Function

The complete PINN objective function is typically

$$
L=
w_dL_{\text{data}}
+
w_pL_{\text{PDE}}
+
w_bL_{\text{BC}}
+
w_iL_{\text{IC}}
$$

where:

- $w_d$ = data weight
- $w_p$ = PDE weight
- $w_b$ = boundary weight
- $w_i$ = initial condition weight

Alternatively:

$$
L=
\sum_k w_k L_k
$$

where each physics constraint contributes a separate loss term.

---

# Inverse PINN Loss Function

For inverse problems, unknown physical parameters are also optimized.

Suppose:

$$
\lambda
=
\{k,D,\alpha,\beta,\ldots\}
$$

contains unknown PDE parameters.

The optimization becomes:

$$
\{\theta,\lambda\}
=
\arg\min L
$$

The loss structure remains identical:

$$
L=
L_{\text{data}}
+
L_{\text{PDE}}
+
L_{\text{BC}}
+
L_{\text{IC}}
$$

but now the gradients update both:

- Neural network weights
- Unknown physical parameters

simultaneously.

---

# Example: Heat Equation PINN

Consider

$$
\rho c_p
\frac{\partial T}{\partial t}
=
k
\frac{\partial^2 T}{\partial x^2}
$$

The PDE residual becomes

$$
R(x,t)
=
\rho c_p
\frac{\partial T_\theta}{\partial t}
-
k
\frac{\partial^2 T_\theta}{\partial x^2}
$$

Physics loss:

$$
L_{\text{PDE}}
=
\frac{1}{N_r}
\sum R^2
$$

Total loss:

$$
L=
w_dL_{\text{data}}
+
w_pL_{\text{PDE}}
+
w_bL_{\text{BC}}
+
w_iL_{\text{IC}}
$$

Training drives the neural network toward a temperature field that simultaneously satisfies:

- Experimental measurements
- Heat-transfer physics
- Initial conditions
- Boundary conditions

---

# Practical Loss Design Considerations

### 1. Loss Balancing

Different loss terms may have vastly different magnitudes.

A common issue is:

$$
L_{\text{PDE}}
\gg
L_{\text{data}}
$$

or

$$
L_{\text{data}}
\gg
L_{\text{PDE}}
$$

leading to poor convergence.

Adaptive weighting methods are often used to balance gradients.

---

### 2. Residual Sampling

The PDE residual is only evaluated at collocation points.

Increasing the number and quality of collocation points generally improves physical consistency.

---

### 3. Hard vs Soft Constraints

**Soft Constraints**

Boundary and initial conditions are added through loss terms:

$$
L_{\text{BC}},\;
L_{\text{IC}}
$$

**Hard Constraints**

The network architecture is modified such that BCs or ICs are satisfied exactly.

Hard constraints usually improve convergence.

---

### 4. Multi-Physics Problems

For coupled physics systems:

$$
L=
L_{\text{PDE}_1}
+
L_{\text{PDE}_2}
+
L_{\text{PDE}_3}
+
L_{\text{BC}}
+
L_{\text{IC}}
+
L_{\text{data}}
$$

Examples:

- Electro-thermal systems
- Fluid-structure interaction
- Battery electrochemistry
- Motor thermal networks
- Electromagnetic-thermal coupling

---

# Key Takeaway

A PINN is fundamentally an optimization problem whose success depends on the design of its loss function. For a general PDE, the loss function combines:

$$
\boxed{
L=
L_{\text{data}}
+
L_{\text{PDE}}
+
L_{\text{BC}}
+
L_{\text{IC}}
}
$$

The PDE residual embeds the governing physics, while data, boundary, and initial condition losses ensure the learned solution remains physically realistic and accurate. The same framework naturally extends to inverse PINNs, where unknown physical parameters are learned alongside the solution field.
