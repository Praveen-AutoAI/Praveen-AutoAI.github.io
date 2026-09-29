---
layout: post
title: "Paths to PINNs: Forward and Inverse Physics-Informed Neural Networks for Small-Data Engineering Problems "
description: "Physics-Informed Neural Networks (PINNs): A Beginner-Friendly Introduction with Engineering Applications"
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning,]
tags: [Data Science, Deep Learning, AI]
math: true
---

## Paths to PINNs: Forward and Inverse Physics-Informed Neural Networks for Small-Data Engineering Problems
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
and accurate. The same framework naturally extends to inverse PINNs, where unknown physical parameters are learned alongside the solution field.


# PINN Loss Function: The Magic Math to embedding Physics and Data into the model

In data-driven machine learning, the loss function is the "teacher" that tells the model what constitutes a good prediction. The model will learn exactly what the loss function encourages and ignore what it does not penalize

### Key Roles of a Loss Function

- **Quantifies Error**  
  Measures the difference between model predictions and ground-truth data, providing a numerical score that indicates prediction quality.

- **Drives Optimization**  
  Provides the feedback signal used by optimization algorithms (e.g., Adam, SGD) to update model parameters and minimize prediction errors.

- **Defines Learning Objectives**  
  Specifies what the model should learn through task-dependent formulations such as MSE/MAE for regression and Cross-Entropy for classification.

- **Incorporates Regularization**  
  Adds penalty terms to control model complexity, reduce overfitting, and encourage better generalization.

- **Controls Generalization and Behavior**  
  Influences model characteristics such as robustness, smoothness, noise sensitivity, and resistance to overfitting, ensuring the model learns patterns rather than memorizing data.

- **Enables Objective Benchmarking**  
  Serves as a common metric for comparing architectures, evaluating performance, and tuning hyperparameters.

- **Transforms Data into Knowledge**  
  Learns the underlying relationship between inputs and outputs, enabling accurate predictions on previously unseen data.

> **In summary:** The loss function acts as the learning objective, optimization guide, performance metric, and behavioral controller of a machine learning model. It ultimately determines what the model learns and how well it generalizes to new data.

The main idea behind a PINN is simple:

> A neural network should not only fit the data, but also obey the laws of physics.

To achieve this, PINNs extend the traditional neural network loss function by adding physics-based penalties.

---

## 1. Traditional Neural Network

A standard neural network only learns from data:

$$ L = L_{data} $$

For example:

$$L_{data}=\frac{1}{N}\sum_{i=1}^{N}(y_{pred,i}-y_{true,i})^2$$

The network simply tries to reduce prediction error.

---

## 2. PINN Idea

Suppose the system is governed by a partial differential equation (PDE):

$$
\mathcal{N}(u)=0
$$

where:

- $u$ = unknown physical solution
- $\mathcal{N}$ = differential operator representing the governing physics

Examples include heat transfer, fluid flow, structural dynamics, and electromagnetics, where the governing laws are expressed through differential equations.

Instead of solving the PDE using traditional numerical methods such as Finite Element Method (FEM) or Finite Difference Method (FDM), a Physics-Informed Neural Network (PINN) approximates the solution using a neural network:

$$
u_\theta(x,t)
$$

where:

- $x,t$ = spatial and temporal coordinates
- $\theta$ = trainable neural network parameters (weights and biases)

The neural network receives the coordinates $(x,t)$ as inputs and predicts the corresponding physical state $u$.

However, obtaining a prediction is not sufficient. The prediction must also satisfy the underlying physical laws. Therefore, the network output is substituted into the governing PDE, and all required derivatives are computed using **automatic differentiation**.

The resulting PDE residual is defined as:

$$
R = \mathcal{N}(u_\theta)
$$

The residual acts as a measure of physics violation. It quantifies how far the neural network prediction deviates from the governing equation at any point in the domain.

If the physical law is satisfied perfectly:

$$
R = 0
$$

For example Heat equation:

$$ \frac{\partial T}{\partial t} - \alpha \frac{\partial^2 T}{\partial x^2} = 0 $$

The neural network predicts:

$$ T_\theta(x,t) $$

Substituting the network prediction into the PDE produces the residual:

$$ R = \frac{\partial T_\theta}{\partial t} - \alpha \frac{\partial^2 T_\theta}{\partial x^2} $$

If the prediction perfectly satisfies the governing physics:

$$ R = 0 $$

> KEY INSIGHT: During training, the PINN minimizes this residual over many collocation points throughout the domain. Consequently, the network is forced not only to fit available data but also to obey the governing physics, allowing accurate learning even when only a small amount of measured data is available.

---

## 3. Physics Loss

The PDE residual is converted into a loss term:

$$L_{physics} = \frac{1}{N} \sum_{i=1}^{N} R_i^2$$

This teaches the network:

> "Do not just fit the data. Also obey the governing equation."

<div style="border-left:4px solid #1E88E5; padding:10px; background:#f4f9ff;">
<strong>Key Idea:</strong><br>
> Do not just fit the data. Also obey the governing equation.
</div>

---

## 4. Total PINN Loss

Now the neural network minimizes both: Total loss includes data and physics

$$ L =L_{data}+\lambda L_{physics}$$

Where:

- $L_{data}$ measures data fitting error  
- $L_{physics}$ measures PDE violation  
- $\lambda$ controls the importance of physics


---

## 5. Adding Boundary Conditions

The loss function can get complex with multiple constraints. Suppose the solution must satisfy:

$$ u(0,t)=100 $$

The boundary-condition loss becomes:

$L_{BC} = \frac{1}{N} \sum_{i=1}^{N} \left(u_{pred,i} - u_{BC,i}\right)^2$

The total loss is now:

$L = L_{data} + L_{physics} + L_{BC}$

---

## 6. Adding Initial Conditions

For transient problems:

$$ u(x,0)=u_0(x) $$

The initial-condition loss becomes:

$L_{IC} = \frac{1}{N} \sum_{i=1}^{N} \left(u_{pred,i} - u_{0,i}\right)^2$

The complete PINN loss becomes:

$L = L_{data} + L_{physics} + L_{BC} + L_{IC}$

---

## Evolution of PINN Loss Functions

### Traditional Neural Network

$$  L = L_{data} $$

**Goal:** Fit the data

---

### Basic PINN

$L = L_{data} + L_{physics}$

**Goal:** Fit the data + obey the PDE

---

### Practical PINN

$L = L_{data} + L_{physics} + L_{BC} + L_{IC}$

**Goal:** Fit the data + obey physics + satisfy boundary conditions + satisfy initial conditions

---

## Training Loop for PINN
<div style="text-align: center;">
  <img src="/assets/images/PINN/I_PINN_5.png" alt="PINN Difference" width="800">
</div>

---

## The Big Picture

A PINN learns from two teachers:

### Teacher 1: Data

$$ L_{data} $$

Experimental measurements tell the network what the solution should look like.

### Teacher 2: Physics

$$ L_{physics} $$

The governing PDE tells the network what solution is physically possible.

---

## Key Takeaway

$\boxed{L = L_{data} + L_{physics} + L_{BC} + L_{IC}}$

A traditional neural network learns only from data.

A Physics-Informed Neural Network (PINN) learns from:

- Data
- Governing PDEs
- Boundary Conditions
- Initial Conditions

| **Aspect** | **Physics-Informed Neural Networks (PINNs)** | **Conventional Data-Driven Machine Learning** |
|------------|---------------------------------------------|----------------------------------------------|
| **Learning Paradigm** | Combines data with governing physical laws. | Learns patterns solely from data. |
| **Physics Knowledge** | Known and explicitly incorporated through ODEs/PDEs. | Typically unknown or ignored. |
| **Data Requirements** | Can work effectively with sparse, limited, or partially observed data. | Usually requires large amounts of labeled training data. |
| **Model Guidance** | Guided by both measurement data and physics-based constraints. | Guided only by prediction error on training data. |
| **Loss Function** | Includes data loss + physics residual loss + boundary/initial condition losses. | Typically includes only data loss (e.g., MSE, MAE, Cross-Entropy). |
| **Generalization** | Improved extrapolation by enforcing physical consistency. | Often struggles outside the training data distribution. |
| **Interpretability** | Predictions are physically meaningful and easier to explain. | Often behaves as a black-box model. |
| **Data Scarcity Performance** | Performs well even when measurements are scarce. | Performance degrades significantly with limited data. |
| **Physical Consistency** | Explicitly constrained to satisfy governing equations. | May generate physically impossible predictions. |
| **Typical Applications** | Heat transfer, fluid dynamics, battery modeling, digital twins, structural mechanics, electromagnetics. | Image recognition, speech processing, recommendation systems, forecasting, pattern recognition. |
| **Computational Cost** | Higher training cost due to PDE residual and automatic differentiation calculations. | Generally lower training complexity. |
| **Primary Objective** | Learn solutions while obeying known physics. | Learn statistical relationships from data. |
| **Best Use Case** | Physics is known but data is limited. | Data is abundant but governing physics is unknown or difficult to model. |

### Key Takeaway

| **Scenario** | **Preferred Approach** |
|--------------|------------------------|
| Known physics + limited measurements | **PINN** |
| Unknown physics + large datasets | **Data-Driven ML** |
| Need physically consistent predictions | **PINN** |
| Pattern recognition and feature extraction | **Data-Driven ML** |
| Parameter estimation from sparse measurements | **Inverse PINN** |
| Pure prediction accuracy with abundant data | **Data-Driven ML** |
