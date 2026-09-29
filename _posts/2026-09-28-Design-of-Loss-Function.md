---
layout: post
title: "Matching Made in Heaven: Pratical Loss Function Design for Engineering Problems "
description: "Matching Made in Heaven: Pratical Loss Function Design for Engineering Problems "
date: 2026-09-07
categories: [Machine Learning, Engineering, Scientific Machine Learning, PINN]
tags: [Data Science, Deep Learning, AI,PINN]
math: true
---

## Matching Made in Heaven: Pratical Loss Function Design for Engineering Problems

### Bridging the two worlds : World of Data & Physics

For decades, engineers have lived in two different worlds.

In one world, we have **physics-based models**. These models are built from first principles, governing equations, and decades of scientific understanding. They are trustworthy, interpretable, and physically meaningful. However, they often become difficult to develop when the system is highly complex or when some parameters are unknown.

In the other world, we have **data-driven machine learning**. Given enough data, neural networks can learn remarkably complex relationships. Yet they suffer from a fundamental weakness:

> *A neural network can fit the data perfectly and still violate the laws of physics.*

This raises an interesting question:

> *What if a neural network could learn from data and simultaneously obey the underlying physics?*

This simple but powerful idea gives birth to **Physics-Informed Neural Networks (PINNs)**.

Unlike conventional neural networks, PINNs are not trained solely to minimize prediction error. Instead, they are trained to satisfy the governing equations of the system as well. The mechanism that enables this is the **loss function**, which can be viewed as the "teacher" guiding the network during training by **penalizing the wrong behavior.**

A typical PINN loss takes the form:

$$L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{BC}} + L_{\text{IC}}$$

where:

- $$L_{\text{data}}$$ ensures agreement with measurements.
- $$L_{\text{physics}}$$ enforces the governing ODEs/PDEs.
- $$L_{\text{BC}}$$ enforces boundary conditions.
- $$L_{\text{IC}}$$ enforces initial conditions.

The real magic of PINNs lies in **loss function design**. By carefully choosing which physical constraints to encode and how strongly to enforce them, we can transform sparse measurements into physically consistent solutions, estimate unknown parameters, and even discover hidden system dynamics. In many ways, the success of a PINN is determined less by the neural network architecture itself and more by how intelligently the loss function captures the physics of the problem.

<p style="color:blue;">
<strong>My Insight:</strong> When we look at the model architecture, training process, usage of data, and so on... I realize that the <strong>PINNs are not breaking the rules of the game, but they bend it.</strong> And the bending is done by the loss function and not the neural networks.
</p>

---

Let us understand the PINN by working on three examples.

| Example | Type | Network Learns | Unknown Quantity |
| :--- | :--- | :--- | :--- |
| Heat Transfer | Forward PINN | $T(x)$ | Temperature Distribution |
| Spring-Mass with Force | Forward PINN | $x(t)$ | Dynamic Response |
| Heat Transfer + Conductivity Identification | Inverse PINN | $T(x)$, $k$ | Thermal Conductivity |



### Example 1: Forward PINN for Steady-State Heat Transfer

### Physical Problem

Let's start with one of the simplest engineering problems: heat conduction through a metal rod.

Imagine a long metal rod whose ends are maintained at fixed temperatures.

```text
x = 0                              x = L
|------------------------------------|
T = 100°C                         T = 0°C
```

The left end is held at **100°C**, while the right end is held at **0°C**.

For simplicity, we assume:

- No heat is generated inside the rod.
- No heat escapes to the surroundings.
- The material properties remain constant.

Under these conditions, heat naturally flows from the hot end to the cold end until the system reaches **steady state**.

Our objective is simple:

> Given the governing physics, can a PINN learn the temperature distribution inside the rod?

---

### Governing Equation

From Fourier's law of heat conduction, the steady-state heat equation for this problem becomes:

$$
\frac{d^2T}{dx^2}=0
$$

where:

- \(T\) = temperature
- \(x\) = position along the rod

At first glance, this equation may look abstract.

A useful way to interpret it is:

> The curvature of the temperature profile is zero everywhere.

If the curvature is zero, the temperature must vary linearly from one end of the rod to the other.

In fact, we already know the exact physical solution should look something like:

```text
100°C |\
      | \
      |  \
      |   \
      |    \
  0°C +-----\----------> x
```

A PINN must discover this temperature profile while simultaneously respecting the physics.

---

### Boundary Conditions

To obtain a unique solution, we must specify the temperatures at the boundaries.

$$
T(0)=100
$$

$$
T(L)=0
$$

These conditions tell us:

- At the left end, temperature is 100°C.
- At the right end, temperature is 0°C.

Think of boundary conditions as anchors that hold the solution in place.

Without them, infinitely many temperature distributions could satisfy the differential equation.

---

### PINN Representation

Instead of solving the differential equation directly, a PINN uses a neural network to approximate the temperature field.

```text
Position x
     ↓
Neural Network
     ↓
Temperature T(x)
```

The neural network acts as a mathematical function:

$$
T(x) \approx T_\theta(x)
$$

where:

- \(T_\theta(x)\) is the neural network prediction
- \(\theta\) represents all trainable weights and biases

Initially, the network predictions are random.

Training gradually adjusts the weights until the predicted temperature field satisfies both the measurements and the governing physics.

---

### PINN Residual

The governing equation requires

$$
\frac{d^2T}{dx^2}=0
$$

Using automatic differentiation, PINNs can calculate derivatives directly from the neural network.

First derivative:

$$
T_x=\frac{dT}{dx}
$$

Second derivative:

$$
T_{xx}=\frac{d^2T}{dx^2}
$$

The heat-equation residual is defined as:

$$
R=T_{xx}
$$

This residual measures how much the neural network violates the governing equation.

If the physics is perfectly satisfied:

$$
R=0
$$

Everywhere inside the rod.

You can think of the residual as a "physics error."

```text
Residual = 0
     ↓
Physics Satisfied

Residual ≠ 0
     ↓
Physics Violated
```

---

### Loss Function Design

This is where the real magic of PINNs happens.

Instead of learning only from data, the network learns from multiple sources of information.

---

| Loss Component | Description & Formula | What it Teaches the Network |
| :--- | :--- | :--- |
| **Data Loss** | $L_{data} = \frac{1}{N} \sum (T_{pred} - T_{true})^2$<br><br>Based on a few temperature measurements from sensors.<br><br>`Sensor Location` $\rightarrow$ `Measured Temperature` | *"Match the temperatures measured in the experiment."*<br><br>*(Without this term, the network might satisfy physics but not match reality.)* |
| **Physics Loss** | $L_{physics} = \frac{1}{M} \sum R^2$<br><br>Where the residual is:<br>$R = T_{xx}$ | *"Even where no measurements exist, obey the heat equation."*<br><br>*(Traditional neural networks learn only where data exists. PINNs learn everywhere because physics applies everywhere.)* |
| **Boundary Condition Loss** | $L_{BC} = (T(0)-100)^2 + (T(L)-0)^2$<br><br>The boundary temperatures must remain fixed. | *"Never violate the known temperatures at the rod boundaries."*<br><br>*(Without this loss, the network could predict physically impossible temperatures at the ends.)* |

<p style="color:blue;">
<strong>My Insight:</strong> Without the physics loss, the neural network might learn the physically correct representation, but the chances are very low . The network uses brute force to map the input to output and less likely to learn the governing law of the system. May be yes if the data is abundant and of good quality.
</p>

---

### Step 4: Total Loss

All three objectives are combined together.

$L_{total} = L_{data} + L_{physics} + L_{BC}$

During training, the optimizer minimizes this total loss.

In practical terms, the network is simultaneously trying to:

1. Match measurements.
2. Obey the heat equation.
3. Respect boundary conditions.

---

### Understanding the Training Process

A useful way to visualize PINN training is:

```text
                 Position x
                      ↓
               Neural Network
                      ↓
                   T(x)
                ↙   ↓   ↘
               ↙    ↓    ↘
          Data   Physics   BC
          Loss     Loss   Loss
               ↘   ↓   ↙
                Total Loss
                      ↓
               Weight Update
                      ↓
             Better T(x)
```

At every training iteration:

1. The network predicts temperature.
2. Automatic differentiation computes derivatives.
3. Physics residual is evaluated.
4. Losses are calculated.
5. Weights are updated.
6. The process repeats.

Eventually, the network discovers a temperature profile that satisfies all requirements.

---

### PyTorch Skeleton

```python
class HeatPINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)
```

---

### What the Neural Network Learns

The neural network learns:

$$
T(x)
$$

which represents the temperature at every location inside the rod.

Instead of predicting temperatures only at sensor locations, it learns a continuous temperature field.

---

### What Physics Contributes

Physics contributes the governing equation:

$$
\frac{d^2T}{dx^2}=0
$$

This equation acts like a built-in engineering supervisor that continuously checks whether the prediction makes physical sense.

Even with very few measurements, physics prevents the network from producing unrealistic temperature profiles.

---

### Example: 3 Inverse PINN for Heat Transfer Parameter Identification

In a traditional heat transfer problem, the thermal conductivity $$k$$ of a material is already known. Engineers use the governing heat equation and boundary conditions to predict the temperature distribution inside the material. This is called a **forward problem**.

An **Inverse Physics-Informed Neural Network (Inverse PINN)** solves the opposite problem. Instead of knowing the conductivity and predicting temperature, we start with a few measured temperature values and use them to estimate the unknown conductivity $$k$$.

Consider a metal rod maintained at $$100^\circ C$$ on one end and $$0^\circ C$$ on the other, with internal heat generation $$q$$. Temperature sensors placed along the rod provide measured temperature data. The objective is to identify the material's thermal conductivity.

The neural network receives the spatial position $$x$$ and predicts temperature $$T(x)$$. Unlike a standard PINN, the conductivity $$k$$ is also treated as a trainable parameter. During training, automatic differentiation computes derivatives such as

$$
\frac{dT}{dx}
$$

and

$$
\frac{d^2T}{dx^2}.
$$

These derivatives are substituted into the governing heat equation:

$$
k\frac{d^2T}{dx^2}+q=0
$$

to form the physics residual:

$$
R = kT_{xx}+q
$$

The network minimizes three losses simultaneously:

- **Data Loss ($$L_{\text{data}}$$)**: matches measured temperatures.
- **Physics Loss ($$L_{\text{physics}}$$)**: enforces the heat equation.
- **Boundary Condition Loss ($$L_{\text{BC}}$$)**: satisfies boundary temperatures.

$$
L_{\text{total}} = L_{\text{data}} + L_{\text{physics}} + L_{\text{BC}}
$$

As training progresses, both the neural network weights and conductivity $$k$$ are updated. Eventually, the predicted temperature curve matches the measurements while also satisfying the governing physics.

The final result is a physically consistent temperature field and an accurate estimate of the unknown thermal conductivity, even when only limited measurement data are available.

### Inverse PINN for Heat Transfer 
![Inverse_PINN](/assets/images/PINN/I_PINN_6.png)


