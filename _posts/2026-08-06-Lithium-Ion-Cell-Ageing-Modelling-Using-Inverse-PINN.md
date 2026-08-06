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

**Objective** is to develop **Inverse Physics-Informed Neural Network (iPINN)** method to accurately recover battery degradation parameters directly from sparse experimental observations while maintaining physical consistency with established battery-ageing theory.

Lithium-ion cells have 2 dimension of ageing a) Cyclic ageing and b) Calendar ageing. While the cell degradation is complex multi-dimensional process, that is not yet fully understood, there are successful mathematical hypothesis to model some of the degradation mechanisms (SEI formation, Lithium plating, Electrode cracking, electrolyte oxidation are some of them)
To demonstrate the idea, potential and beauty of PINN/iPINN, here I am considering calendar ageing dataset, where the capacity loss as a function of temperature.

<h3>Dataset</h3>

<div style="font-size: 16px;
border-left: 5px solid #2e86de;
padding: 12px 18px;
background-color: #f8f9fa;
border-radius: 4px;
">

The experimental data used in this **Inverse Physics-Informed Neural Network (iPINN)** demonstration corresponds to calendar-ageing measurements of a commercial **Lithium Iron Phosphate (LFP)** cell. The objective of the original study was to investigate the impact of **storage temperature** on battery degradation under high State-of-Charge (SoC) conditions.

### Cell Specifications

The experiments were performed on a commercial **Sony US26650FTC1** cell designed for stationary energy-storage applications.

#### Cell Characteristics

- **Format:** 26650 Cylindrical Cell
- **Chemistry:** Lithium Iron Phosphate (LiFePO₄)
- **Nominal Capacity:** 3000 mAh
- **Nominal Voltage:** 3.2 V
- **Operating Voltage Range:** 2.0 V to 3.6 V

---

### Calendar Ageing Test Conditions

- **Storage State:** Constant 100% State of Charge (SoC)
- **Temperature Matrix:** 15°C, 25°C, 35°C, and 45°C
- **Time Span:** Day 0 to approximately Day 235 (sparse observation checkpoints)

The original study reports **capacity loss** over time. For the iPINN implementation, the data was converted to **remaining capacity**, which serves as the target variable for the neural network:

```text
Q = 1 - Loss
```

where:

- **Q** = Remaining normalized capacity
- **Loss** = Measured capacity fade

Using remaining capacity instead of capacity loss allows the network to directly model battery **State-of-Health (SoH)** evolution while simultaneously enforcing the underlying degradation physics through the PINN framework.

### Experimental Data (Impact of Temperature on Calendar Ageing)
![Impact of Temperature on Calendar Ageing](/assets/images/Experimental_Data.jpg)

---

### Source

The experimental data was extracted from:

> Naumann, M. et al., *Analysis and Modeling of Calendar Aging of a Commercial LiFePO₄/Graphite Cell*, Journal of The Electrochemical Society.

📄 **Paper:**  
https://iopscience.iop.org/article/10.1149/2.1181714jes


<h3>Physics Model</h3>

<div style="font-size: 16px;
border-left: 5px solid #2e86de;
padding: 12px 18px;
background-color: #f8f9fa;
border-radius: 4px;
">

## Physics Model

The degradation of lithium-ion batteries is a complex multi-dimensional phenomenon involving several interacting ageing mechanisms. While many degradation pathways exist, calendar ageing is largely governed by **SEI (Solid Electrolyte Interphase) growth** and long-term parasitic side reactions occurring at the electrode-electrolyte interface.

The objective of this work is not to model every degradation mechanism, but rather to demonstrate how an **Inverse Physics-Informed Neural Network (iPINN)** can identify physically meaningful degradation parameters directly from sparse experimental observations.

---

### Step 1: Capacity Loss Model

Battery capacity loss is assumed to arise from two degradation mechanisms:

- Diffusion-limited SEI growth
- Long-term parasitic reactions

The capacity-loss model is:

```text
Loss(t) = a·√t + b·t
```

where:

- **a** = SEI-growth contribution
- **b** = long-term ageing contribution
- **t** = storage time

The √t term dominates early-life ageing, while the linear term captures slow continuous degradation.

---

### Step 2: Define Remaining Capacity

The PINN is trained on normalized remaining capacity:

```text
Q(t) = 1 − Loss(t)
```

Substituting the capacity-loss model:

```text
Q(t) = 1 − a·√t − b·t
```

---

### Step 3: Differentiate with Respect to Time

Taking the derivative:

```text
dQ/dt = −(0.5·a·t⁻⁰·⁵ + b)
```

Defining:

```text
k = 0.5·a
```

gives:

```text
dQ/dt = −(k·t⁻⁰·⁵ + b)
```

---

### Step 4: Governing Equation

Rearranging gives the governing degradation equation:

```text
dQ/dt + k·t⁻⁰·⁵ + b = 0
```

---

### Step 5: Physical Interpretation

The governing equation contains two physically meaningful ageing mechanisms:

```text
dQ/dt + k·t⁻⁰·⁵ + b = 0
```

where:

- **k·t⁻⁰·⁵** → Diffusion-limited SEI growth
- **b** → Long-term parasitic ageing
- **dQ/dt** → Instantaneous capacity-fade rate

The model naturally predicts that SEI-driven degradation slows down with time because the term **t⁻⁰·⁵** decreases as storage time increases.

---

### Step 6: Why PINNs Use the Differential Equation

Instead of enforcing the integrated solution:

```text
Loss(t) = a·√t + b·t
```

PINNs enforce the local governing physics:

```text
R(t) = dQ_PINN/dt + k·t⁻⁰·⁵ + b
```

The corresponding physics loss is:

```text
Loss_Physics = MSE(R(t))
```

By minimizing this residual, the PINN learns a degradation trajectory that not only matches the measured data but also obeys the governing physics throughout the entire time domain.

The iPINN therefore learns:

- Capacity trajectory Q(t)
- SEI-growth parameter k
- Linear-ageing parameter b

directly from experimental observations.

---

## PINN Loss Function

The overall loss formulation should answer three key questions:

✔ Does the model match the measured capacity fade?  
✔ Does the model obey the governing physics?  
✔ Does the model start from the correct battery state?

---

### Singularity-Free Formulation

The original governing equation is:

```text
dQ/dt + k·t⁻⁰·⁵ + b = 0
```

The term **t⁻⁰·⁵** creates a singularity near t = 0.

To improve numerical stability, the equation is reformulated as:

```text
√t·dQ/dt + k + b·√t = 0
```

Both equations represent the same physics, but the second form is significantly more stable for neural-network optimization.

---

### Physics Residual

The PINN residual becomes:

```text
R(t) = √t·dQ_PINN/dt + k + b·√t
```

and the physics loss is:

```text
Loss_Physics = MSE(R(t))
```

---

### Data Loss

The data loss ensures agreement with the experimental measurements:

```text
Loss_Data = MSE(Q_PINN , Q_Data)
```

---

### Initial Condition Loss

The battery starts at full normalized capacity:

```text
Q(0) = 1
```

This constraint is enforced through:

```text
Loss_IC = MSE(Q_PINN(0), 1)
```

---

### Total Loss Function

The overall training objective becomes:

```text
Loss_Total
=
Loss_Data
+
λ_Physics · Loss_Physics
+
λ_IC · Loss_IC
```

where:

- **λ_Physics** controls the importance of satisfying the governing physics.
- **λ_IC** controls the strength of the initial-condition constraint.

This formulation ensures that the network simultaneously:

- Fits the measured battery-ageing data
- Satisfies the governing degradation physics
- Starts from the correct initial condition

while discovering physically meaningful degradation parameters directly from sparse experimental observations.
