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

# Example 2: Forward PINN for a Spring-Mass System with External Force

## Physical Problem

Let's move from a thermal system to a dynamic mechanical system.

Consider a mass attached to a spring that is subjected to an external force.

This is one of the most fundamental systems in engineering and appears in many applications:

- Vehicle suspensions, Vibration isolators, Machine foundations, Structural dynamics, Robotic actuators

Suppose we know:

- Mass \(m\)
- Spring stiffness \(k\)
- External force \(F(t)\)

Our objective is:

> Predict how the mass moves over time, i.e., determine the displacement \(x(t)\).

---

## Understanding the Physics

Whenever a force acts on a mass, Newton's Second Law governs the motion.

The external force tries to move the mass.

The spring resists that movement and attempts to pull the mass back toward its equilibrium position.

The balance between these forces determines the displacement response.

```text
External Force
      ↓
Moves Mass
      ↓
Spring Generates Restoring Force
      ↓
Motion Over Time
```

---

## Governing Equation

Newton's Second Law gives:

$$
m\ddot{x}+kx=F(t)
$$

where:

- \(x\) = displacement
- \(\dot{x}\) = velocity
- \(\ddot{x}\) = acceleration
- \(m\) = mass
- \(k\) = spring stiffness
- \(F(t)\) = applied external force

Before proceeding, let's build some intuition.

### Spring Force

The spring force is

$$
kx
$$

which increases as the spring stretches.

### Inertia Force

The inertia force is represented by

$$
m\ddot{x}
$$

which describes the tendency of the mass to resist acceleration.

Therefore, the equation

$$
m\ddot{x}+kx=F(t)
$$

simply means:

> The external force must balance the inertia force and the spring restoring force.

---

## Initial Conditions

To uniquely determine the motion, we must know the initial state of the system.

Initial displacement:

$$
x(0)=x_0
$$

Initial velocity:

$$
\dot{x}(0)=v_0
$$

These conditions tell us:

- Where the mass starts.
- How fast it is moving at the beginning.

Without these conditions, many solutions could satisfy the governing equation.

---

## PINN Representation

Instead of solving the differential equation directly, we use a neural network.

```text
Time t
   ↓
Neural Network
   ↓
Displacement x(t)
```

Mathematically,

$$
x(t)\approx x_\theta(t)
$$

where:

- \(x_\theta(t)\) is the network prediction
- \(\theta\) represents the trainable parameters of the neural network

Initially, the predictions are random.

Training gradually adjusts the network until the predicted motion satisfies both measurements and physics.

---

## Automatic Differentiation

One of the superpowers of PINNs is automatic differentiation.

Since the network predicts displacement,

$$
x(t)
$$

we can automatically obtain velocity:

$$
\dot{x}
=
\frac{dx}{dt}
$$

and acceleration:

$$
\ddot{x}
=
\frac{d^2x}{dt^2}
$$

without using numerical differentiation.

```text
Neural Network
      ↓
     x(t)
      ↓
 Automatic Differentiation
      ↓
   Velocity
      ↓
 Acceleration
```

This capability allows PINNs to directly enforce differential equations during training.

---

## PINN Residual

The governing equation should be satisfied everywhere.

Recall:

$$
m\ddot{x}+kx=F(t)
$$

The PINN therefore computes a residual:

$$
R
=
m\ddot{x}
+
kx
-
F(t)
$$

Think of this residual as a physics error.

If the prediction perfectly satisfies Newton's law:

$$
R=0
$$

At every time instant.

```text
R = 0
   ↓
Physics Satisfied

R ≠ 0
   ↓
Physics Violated
```

The objective of PINN training is to make this residual as close to zero as possible.

---

# Loss Function Design

This is where PINNs become fundamentally different from conventional neural networks.

Instead of learning only from measurements, they learn from both measurements and governing physics.

---

## Step 1: Data Loss

Suppose a sensor measures displacement at a few time instances.

```text
Time
 ↓
Measured Displacement
```

The network prediction should match these measurements.

The data loss is:

$$
L_{data}
=
\frac{1}{N}
\sum
(x_{pred}-x_{true})^2
$$

### What is this teaching the network?

It tells the network:

> "Match the measured motion of the system."

Without this term, the network may satisfy physics but fail to reproduce the actual measured behavior.

---

## Step 2: Physics Loss

The residual is:

$$
R
=
m\ddot{x}
+
kx
-
F(t)
$$

Physics loss becomes:

$$
L_{physics}
=
\frac{1}{M}
\sum R^2
$$

### What is this teaching the network?

It tells the network:

> "No matter what displacement you predict, it must obey Newton's Second Law."

This is the term that injects engineering knowledge directly into the learning process.

---

## Step 3: Initial Condition Loss

The system must start from the correct initial state.

Therefore:

$$
L_{IC}
=
(x(0)-x_0)^2
+
(\dot{x}(0)-v_0)^2
$$

### What is this teaching the network?

It tells the network:

> "Begin the simulation from the correct displacement and velocity."

Without this term, the network might predict the correct shape of the response while starting from the wrong physical state.

---

## Step 4: Total Loss

All objectives are combined together.

$$
L_{total}
=
L_{data}
+
L_{physics}
+
L_{IC}
$$

The optimizer continuously updates the network weights until this total loss becomes as small as possible.

---

## Understanding the Training Process

A useful picture of PINN training is:

```text
                    Time t
                       ↓
                Neural Network
                       ↓
                     x(t)
                       ↓
        Automatic Differentiation
                 ↓           ↓
              ẋ(t)       ẍ(t)
                   \      /
                    \    /
               Physics Residual
                       ↓
                Physics Loss

Measured Data ───────→ Data Loss

Initial Conditions ─→ IC Loss

        Data Loss
             +
        Physics Loss
             +
          IC Loss
             ↓
        Total Loss
             ↓
      Weight Update
             ↓
       Better x(t)
```

---

## PyTorch Skeleton

```python
class SpringPINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,t):
        return self.net(t)
```

---

## Expected Solution

After training, the network learns the displacement response.

A typical response may look like:

```text
Displacement

 ^
 |\
 | \
 |  \
 |   \__
 |      \__
 |          \__
 +-----------------> Time
```

The displacement changes continuously while respecting the governing dynamics.

---

## What the Neural Network Learns

The neural network learns:

$$
x(t)
$$

which represents the displacement at every instant of time.

Rather than learning only a few measured points, it learns the entire motion history.

---

## What Physics Contributes

Physics contributes Newton's Second Law:

$$
m\ddot{x}+kx=F(t)
$$

This equation acts as an engineering constraint that continuously checks whether the predicted motion is physically realistic.

Even if only a handful of measurements are available, the governing equation helps fill in the gaps.

---

## Key Takeaway

> A traditional neural network learns motion from data alone. A PINN learns motion from both data and Newton's laws.

For this spring-mass system:

```text
Measured Displacement
           +
External Force
           +
Newton's Law
           +
Initial Conditions
           ↓
          PINN
           ↓
Physically Consistent Motion
```

The most important lesson from this example is that the neural network is not simply fitting displacement measurements. It is learning a motion trajectory that simultaneously satisfies experimental observations and the underlying laws of mechanics.

As a result, PINNs can often learn accurate and physically meaningful solutions from far fewer measurements than a conventional neural network, making them particularly attractive for engineering applications where data is scarce but physical laws are well understood.
