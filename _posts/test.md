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

# PINN Loss Function Design: A Beginner's View

The fundamental idea behind a Physics-Informed Neural Network (PINN) is remarkably simple:

> A neural network should not only fit the available data, but also obey the laws of physics.

Traditional neural networks learn solely from observations. If sufficient data is available, they can achieve excellent predictions. However, when data is sparse, noisy, or expensive to collect, traditional models often struggle to generalize.

PINNs address this challenge by introducing an additional source of knowledge: the governing physics of the system.

Instead of asking:

> "How well does the model fit the measurements?"

PINNs ask:

> "How well does the model fit the measurements while simultaneously obeying the governing physical laws?"

This idea is implemented through the loss function.

---

## 1. Traditional Neural Network

A standard neural network learns only from data.

Its objective is typically to minimize the prediction error:

$$
L = L_{data}
$$

where

$$
L_{data}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(y_{pred,i}-y_{true,i}\right)^2
$$

Here:

- $y_{true}$ are the measured values
- $y_{pred}$ are the network predictions

During training, the model repeatedly adjusts its weights to reduce this error.

### What is the limitation?

Imagine we have only five temperature sensors placed along a heated rod.

A conventional neural network can learn those five points, but nothing prevents it from producing physically unrealistic temperatures between the sensors.

In other words:

> Traditional neural networks can fit the data without understanding the physics.

---

## 2. The PINN Idea

Now suppose the system is governed by a known Partial Differential Equation (PDE):

$$
\mathcal{N}(u)=0
$$

where:

- $u$ is the unknown solution field
- $\mathcal{N}$ is the governing differential operator

Examples include:

- Heat equation
- Wave equation
- Diffusion equation
- Navier-Stokes equations

The neural network predicts:

$$
u_\theta(x,t)
$$

using the spatial location and time as inputs.

Unlike a traditional neural network, we now have additional information:

> We know what equations the solution must satisfy.

Therefore, if the neural network violates the PDE, we should penalize it.

---

## 3. Physics Residual: Measuring PDE Violation

To determine whether the prediction obeys the governing physics, we substitute the neural-network prediction directly into the PDE.

This produces a residual:

$$
R = \mathcal{N}(u_\theta)
$$

Think of the residual as a "physics error."

If the prediction perfectly satisfies the governing equation:

$$
R = 0
$$

If the prediction violates physics:

$$
R \neq 0
$$

The larger the residual, the greater the violation of the governing laws.

### Intuition

Consider the heat equation.

If the predicted temperature field does not satisfy heat conservation, the residual becomes large.

A large residual tells the network:

> "Your prediction may fit the measurements, but it does not obey the physics."

---

## 4. Physics Loss

The residual is converted into a loss term:

$$
L_{physics}
=
\frac{1}{N}
\sum_{i=1}^{N}
R_i^2
$$

The purpose of this term is to reward physically consistent solutions and penalize physically impossible ones.

### Conceptually

The data loss asks:

> "Did you match the measurements?"

The physics loss asks:

> "Did you obey the governing equations?"

Both questions are equally important.

---

## 5. Combining Data and Physics

A PINN learns by minimizing both objectives:

$$
L
=
L_{data}
+
\lambda L_{physics}
$$

where:

- $L_{data}$ measures mismatch with observations
- $L_{physics}$ measures violation of the PDE
- $\lambda$ controls the importance of physics

The network must now satisfy two teachers:

### Teacher 1: Data

Experimental measurements describe what was observed.

### Teacher 2: Physics

The governing equations describe what is physically possible.

A valid solution must satisfy both.

---

## 6. Why Boundary Conditions Matter

Most engineering problems also have known boundary conditions.

For example, a heated rod may have:

$$
u(0,t)=100
$$

meaning the left end is maintained at 100°C.

Without enforcing this condition, the network could produce a solution that satisfies the PDE but violates the known boundary behavior.

To prevent this, a boundary-condition loss is added:

$$
L_{BC}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
u_{pred,i}
-
u_{BC,i}
\right)^2
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

### Intuition

The PDE governs what happens inside the domain.

Boundary conditions govern what happens at the edges.

Both are required to obtain a physically meaningful solution.

---

## 7. Why Initial Conditions Matter

For transient problems, the starting state is also known.

For example:

$$
u(x,0)=u_0(x)
$$

A transient simulation without an initial condition is similar to watching a movie from the middle without knowing how it started.

Therefore, we add:

$$
L_{IC}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
u_{pred,i}
-
u_{0,i}
\right)^2
$$

The full PINN objective becomes:

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

Now the solution must satisfy:

- Experimental observations
- Governing equations
- Boundary conditions
- Initial conditions

simultaneously.

---

## Evolution of the PINN Loss Function

### Traditional Neural Network

$$
L = L_{data}
$$

**Goal:** Match the observations.

---

### Basic PINN

$$
L
=
L_{data}
+
L_{physics}
$$

**Goal:** Match the observations and obey the PDE.

---

### Practical Engineering PINN

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

**Goal:** Match the observations while satisfying all known physical constraints.

---

## The Big Picture

A useful way to think about PINNs is that they learn from both experiments and science.

| Source of Knowledge | What It Tells the Network |
|--------------------|---------------------------|
| Data Loss | What was observed |
| Physics Loss | What is physically possible |
| Boundary Loss | How the system behaves at its boundaries |
| Initial Loss | Where the system started |

A traditional neural network sees only the measurements.

A PINN sees both the measurements and the underlying physics.

---

## Key Takeaway

$$
\boxed{
L =
L_{data}
+
L_{physics}
+
L_{BC}
+
L_{IC}
}
$$

The power of a PINN comes from augmenting the traditional data-fitting loss with physics-based constraints.

As a result, PINNs can often learn accurate and physically meaningful solutions from far fewer measurements than a conventional neural network, making them particularly attractive for engineering applications where data is scarce but physical laws are well understood.
