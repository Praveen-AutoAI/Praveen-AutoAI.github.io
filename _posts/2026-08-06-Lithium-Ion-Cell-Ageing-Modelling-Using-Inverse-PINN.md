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

The degradation of lithium-ion batteries is a complex, multi-dimensional phenomenon involving several interacting ageing mechanisms. While numerous degradation pathways exist, calendar ageing is primarily driven by **SEI (Solid Electrolyte Interphase) growth** and other parasitic side reactions occurring at the electrode-electrolyte interface.

The objective of this work is not to develop a comprehensive battery degradation model, but rather to demonstrate how an **Inverse Physics-Informed Neural Network (iPINN)** can identify physically meaningful degradation parameters directly from sparse experimental observations.

To keep the physics simple and interpretable, we assume that battery capacity loss arises from two mechanisms:

- Diffusion-limited SEI growth
- Long-term parasitic ageing reactions

The derivation of the governing equation is presented below.

---

### Step 1: Capacity-Loss Model

The battery capacity loss is represented using a semi-empirical ageing model:

$$
Loss(t) = a\sqrt{t} + bt
$$

where:

- \(a\) = degradation due to SEI growth
- \(b\) = degradation due to long-term ageing reactions
- \(t\) = storage time

The \(\sqrt{t}\) term captures the rapid initial degradation associated with SEI growth, while the linear term represents slower continuous ageing mechanisms.

---

### Step 2: Define Remaining Capacity

The iPINN is trained on **normalized remaining capacity** rather than capacity loss.

$$
Q(t)=1-Loss(t)
$$

Substituting the capacity-loss model:

$$
Q(t)=1-a\sqrt{t}-bt
$$

---

### Step 3: Differentiate with Respect to Time

PINNs enforce governing differential equations, so we require the rate of change of capacity.

Differentiating with respect to time:

$$
\frac{dQ}{dt}
=
-\frac{d}{dt}
\left(
a\sqrt{t}+bt
\right)
$$

Since

$$
\frac{d}{dt}
\left(
\sqrt{t}
\right)
=
\frac{1}{2}t^{-0.5}
$$

we obtain

$$
\frac{dQ}{dt}
=
-\left(
0.5a\,t^{-0.5}
+
b
\right)
$$

---

### Step 4: Rearranging into a Governing Equation

Defining

$$
k = 0.5a
$$

gives

$$
\frac{dQ}{dt}
+
k\,t^{-0.5}
+
b
=
0
$$

This differential equation will be used as the governing degradation law for the PINN.

---

### Step 5: Physical Interpretation

The governing equation contains two physically meaningful degradation mechanisms:

$$
\frac{dQ}{dt}
+
\underbrace{k\,t^{-0.5}}_{\text{Diffusion-Limited SEI Growth}}
+
\underbrace{b}_{\text{Long-Term Linear Ageing}}
=
0
$$

where:

- **\(k\,t^{-0.5}\)** represents degradation due to diffusion-controlled SEI growth.
- **\(b\)** represents degradation due to long-term parasitic reactions.
- **\(\frac{dQ}{dt}\)** represents the instantaneous capacity-fade rate.

The model naturally predicts that SEI-driven degradation slows with time because the term \(t^{-0.5}\) decreases as storage time increases.

---

### Step 6: Why PINNs Use the Differential Equation

Rather than enforcing the integrated solution

$$
Loss(t)=a\sqrt{t}+bt
$$

PINNs enforce the governing differential equation directly.

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

and the corresponding physics loss is

$$
L_{Physics}
=
\mathrm{MSE}
\left(
R(t)
\right)
$$

By minimizing this residual, the PINN learns a solution that not only matches the experimental data but also obeys the governing degradation physics throughout the entire time domain.

The iPINN therefore learns:

- The degradation trajectory \(Q(t)\)
- The SEI-growth parameter \(k\)
- The linear-ageing parameter \(b\)

directly from sparse experimental observations.

---

## PINN Loss Function

The overall loss formulation should answer three key questions:

- Does the model match the measured capacity fade?
- Does the model obey the governing physics?
- Does the model start from the correct battery state?

---

### Singularity Issue and Reformulation

The original governing equation contains the term \(t^{-0.5}\), which becomes singular near \(t=0\):

$$
\frac{dQ}{dt}
+
k\,t^{-0.5}
+
b
=
0
$$

To improve numerical stability, the equation is reformulated by multiplying through by \(\sqrt{t}\):

$$
\frac{dQ}{dt}
+
k\,t^{-0.5}
+
b
=
0
$$

$$
\Downarrow
$$

$$
\sqrt{t}\frac{dQ}{dt}
+
k
+
b\sqrt{t}
=
0
$$

The reformulated equation preserves the underlying physics while removing the singularity at the beginning of the time domain.

---

### Physics Loss

Using the singularity-free formulation, the residual becomes

$$
R(t)
=
\sqrt{t}
\frac{dQ_{PINN}}{dt}
+
k
+
b\sqrt{t}
$$

The physics loss is then

$$
L_{Physics}
=
\mathrm{MSE}
\left(
R(t)
\right)
$$

which enforces the governing degradation physics.

---

### Data Loss

The data loss ensures agreement with the experimental measurements:

$$
L_{Data}
=
\mathrm{MSE}
\left(
Q_{PINN},
Q_{Data}
\right)
$$

---

### Initial-Condition Loss

The battery starts at full normalized capacity:

$$
Q(0)=1
$$

The corresponding initial-condition loss is:

$$
L_{IC}
=
\mathrm{MSE}
\left(
Q_{PINN}(0),
1
\right)
$$

which anchors the solution at the correct initial battery state.

---

### Total Loss Function

The iPINN is trained by simultaneously minimizing the data mismatch, the physics residual, and the initial-condition constraint.

$$
L_{Total}
=
\lambda_{Data}L_{Data}
+
\lambda_{Physics}L_{Physics}
+
\lambda_{IC}L_{IC}
$$

where:

- \(\lambda_{Physics}\) controls the importance of satisfying the governing physics.
- \(\lambda_{IC}\) controls the strength of the initial-condition constraint.

This formulation ensures that the network simultaneously:

- Fits the measured battery-ageing data,
- Satisfies the governing degradation physics,
- Starts from the correct initial condition,

while discovering physically meaningful degradation parameters directly from sparse experimental observations.
