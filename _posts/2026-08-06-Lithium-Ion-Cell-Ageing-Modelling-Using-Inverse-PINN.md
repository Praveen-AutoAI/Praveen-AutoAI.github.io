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

The degradation of lithium-ion batteries is a complex multi-dimensional phenomenon involving several interacting ageing mechanisms. While many degradation pathways exist, calendar ageing is primarily governed by **SEI (Solid Electrolyte Interphase) growth** and other parasitic side reactions occurring at the electrode-electrolyte interface.

The objective of this work is not to develop a complete ageing model covering all degradation mechanisms. Instead, it aims to demonstrate how an **Inverse Physics-Informed Neural Network (iPINN)** can be used to identify temperature-dependent degradation parameters directly from sparse experimental observations.

A richer dataset containing variations in temperature, SoC, depth-of-discharge, current rates, and other operating conditions would allow the formulation of a more comprehensive governing equation. However, for this demonstration, a simplified yet physically meaningful degradation model is adopted.

---

### Step 1: Start from the Capacity Loss Model

Battery capacity loss is assumed to arise from two degradation mechanisms:

- Diffusion-limited SEI growth
- Long-term parasitic reactions

This leads to the semi-empirical ageing model:

$$
Loss(t) = a\sqrt{t} + bt
$$

where:

- \(a\) = degradation due to SEI growth
- \(b\) = degradation due to long-term ageing reactions
- \(t\) = storage time

The first term dominates early-life ageing, while the second term represents slower, continuous degradation.

---

### Step 2: Define Remaining Capacity

The PINN is trained on **normalized remaining capacity** rather than capacity loss.

$$
Q(t)=1-Loss(t)
$$

Substituting the ageing model:

$$
Q(t)=1-a\sqrt{t}-bt
$$

---

### Step 3: Differentiate with Respect to Time

PINNs enforce the governing differential equation, so we require the rate of change of capacity.

Taking the derivative:

$$
\frac{dQ}{dt}
=
-\frac{d}{dt}\left(a\sqrt{t}+bt\right)
$$

Since

$$
\frac{d}{dt}\left(\sqrt{t}\right)
=
\frac{1}{2}t^{-0.5}
$$

we obtain

$$
\frac{dQ}{dt}
=
-\left(0.5a\,t^{-0.5}+b\right)
$$

---

### Step 4: Rearranging into a Governing Equation

Defining

$$
k = 0.5a
$$

gives

$$
\frac{dQ}{dt} + k\,t^{-0.5} + b = 0
$$

---

### Step 5: Physical Interpretation

The resulting differential equation contains two physically meaningful degradation mechanisms:

$$
\frac{dQ}{dt}
+
\underbrace{k\,t^{-0.5}}_{\text{SEI Growth}}
+
\underbrace{b}_{\text{Linear Ageing}}
=
0
$$

where:

- \(k\,t^{-0.5}\) represents diffusion-controlled SEI growth.
- \(b\) represents degradation due to long-term parasitic reactions.
- \(\frac{dQ}{dt}\) represents the instantaneous capacity-fade rate.

The equation naturally predicts that degradation slows down over time because the term \(t^{-0.5}\) decreases as storage time increases.

---

### Step 6: Why PINNs Use the Differential Equation Instead of the Integrated Solution

PINNs enforce the governing differential equation directly rather than the integrated solution.

The physics residual is defined as

$$
R(t)
=
\frac{dQ_{PINN}}{dt}
+
k\,t^{-0.5}
+
b
$$

and the physics loss becomes

$$
Loss_{Physics}
=
MSE\big(R(t)\big)
$$

During training, the network minimizes this residual together with the data loss.

This ensures that the neural network not only fits the experimental measurements but also satisfies the physical constraints embedded in the governing equation.

The iPINN therefore learns:

- The degradation trajectory \(Q(t)\)
- The SEI-growth parameter \(k\)
- The linear-ageing parameter \(b\)

directly from experimental battery-ageing data.

---

## PINN Loss Function

The overall loss function should answer three key questions:

- **Does the model match the measured capacity fade?** → Data Loss
- **Does the model obey the degradation physics?** → Physics Loss
- **Does the model start from the correct battery state?** → Initial Condition Loss

---

### Removing the Singularity

The original governing equation contains the singular term \(t^{-0.5}\):

$$
\frac{dQ}{dt} + k\,t^{-0.5} + b = 0
$$

To improve numerical stability, the equation is reformulated by multiplying through by \(\sqrt{t}\):

$$
\frac{dQ}{dt} + k\,t^{-0.5} + b = 0
$$

⬇️

$$
\sqrt{t}\frac{dQ}{dt} + k + b\sqrt{t} = 0
$$

This singularity-free formulation preserves the original physics while improving optimization stability.

---

### Physics Residual

Using the reformulated equation, the residual becomes

$$
R(t)
=
\sqrt{t}\frac{dQ_{PINN}}{dt}
+
k
+
b\sqrt{t}
$$

The corresponding physics loss is

$$
Loss_{Physics}
=
MSE\big(R(t)\big)
$$

which enforces the governing degradation physics.

---

### Data Loss

The data loss ensures agreement with the experimental observations:

$$
Loss_{Data}
=
MSE
\left(
Q_{PINN},
Q_{Data}
\right)
$$

---

### Initial Condition Loss

Since the battery starts at full normalized capacity,

$$
Q(0)=1
$$

the initial-condition loss is defined as

$$
Loss_{IC}
=
MSE
\left(
Q_{PINN}(0),
1
\right)
$$

This anchors the solution at the correct initial battery state.

---

### Total Loss Function

The overall training objective is

$$
Loss_{Total}
=
Loss_{Data}
+
\lambda_{Phys}Loss_{Physics}
+
\lambda_{IC}Loss_{IC}
$$

where:

- \(\lambda_{Phys}\) controls the importance of satisfying the governing physics.
- \(\lambda_{IC}\) controls the strength of the initial-condition constraint.

This formulation ensures that the network simultaneously:

- Fits the experimental measurements
- Satisfies the governing battery-degradation physics
- Starts from the correct initial condition

while discovering physically meaningful degradation parameters directly from sparse experimental data.
