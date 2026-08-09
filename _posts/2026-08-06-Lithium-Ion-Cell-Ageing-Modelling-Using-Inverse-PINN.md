---
layout: post
title: "Discovering Lithium-Ion Cell Calendar-Ageing Rates Using Inverse Physics-Informed Neural Networks (iPINNs)"
date: 2026-08-04
categories: [Machine Learning, Engineering]
tags: [PINN, Physics, Deep Learning, AI]
math: true
---

  
### Introduction

#### What are Inverse PINNs (iPINNs)?

> A class of **Scientific Machine Learning (SciML)** that embeds governing equations (ODEs/PDEs) into the loss function to infer unknown parameters from data.

- **Forward PINNs:** Given parameters and equations → solve for **state variables**.
- **Inverse PINNs:** Given physical laws and sparse/noisy data → estimate **unknown parameters and degradation rates**.

---

Inverse Physics-Informed Neural Networks (inverse PINNs) refer to a class of **Scientific Machine Learning (SciML)** for inferring unknown parameters, fields, or source terms in differential equation-governed systems by embedding physical knowledge into the loss function of neural network surrogates.

Unlike forward PINNs, which solve for state variables given equations and parameters, inverse PINNs leverage physical laws and sparse or noisy observations to estimate unknown constitutive parameters, source terms, latent fields, or structural design variables.

This class of methods is foundational for scientific machine learning, enabling data-efficient and physically consistent solutions to inverse problems in systems modeled by ordinary and partial differential equations.

#### Key Capabilities and Usage

- **Parameter Recovery & Inference** *(e.g., calendar ageing kinetic parameters)*
- **System Identification** *(discovering governing terms/fields)*
- **Robustness with Sparse and Noisy Data**
- **Uncertainty Quantification**

### Problem Statement

> Battery calendar ageing is strongly influenced by storage temperature. Engineers often require degradation-rate parameters to:
>
> - Predict and forecast battery lifetime
> - Understand underlying ageing mechanisms
> - Compare battery performance across different operating conditions

Traditional machine learning approaches typically require large amounts of data and are often criticized as **black-box models** with limited physical interpretability. Conversely, classical curve-fitting methods can reproduce experimental observations but do not explicitly enforce the underlying physics governing the degradation process.

The **objective** of this work is to develop an **Inverse Physics-Informed Neural Network (iPINN)** framework capable of accurately recovering battery degradation parameters directly from sparse experimental observations while maintaining consistency with established battery-ageing theories.

Lithium-ion cell ageing can generally be categorized into two dimensions:

1. **Calendar Ageing**: Degradation occurring during storage, even when the battery is not cycled.
2. **Cyclic Ageing**: Degradation resulting from charge-discharge cycling.

Although battery degradation is a complex, multidimensional process that is not yet fully understood, several successful mathematical hypotheses have been proposed to describe key degradation mechanisms, including:

- Solid Electrolyte Interphase (**SEI**) growth
- Lithium plating
- Electrode cracking
- Electrolyte oxidation

To demonstrate the concept, potential, and elegance of **PINNs** and **inverse PINNs**, this study focuses on a **calendar-ageing dataset**, where battery capacity loss is analyzed as a function of storage temperature.

The goal is to infer physically meaningful degradation-rate parameters from limited observations(data) while leveraging known governing equations to constrain the learning process.


### Dataset

The experimental data used in this **Inverse Physics-Informed Neural Network (iPINN)** demonstration corresponds to calendar-ageing measurements of a commercial **Lithium Iron Phosphate (LFP)** cell. The objective of the original study was to investigate the impact of **storage temperature** on battery degradation under high State-of-Charge (SoC) conditions.

#### Cell Specifications

The experiments were performed on a commercial **Sony US26650FTC1** cell designed for stationary energy-storage applications.

##### Cell Characteristics

- **Format:** 26650 Cylindrical Cell
- **Chemistry:** Lithium Iron Phosphate (LiFePO₄)
- **Nominal Capacity:** 3000 mAh
- **Nominal Voltage:** 3.2 V
- **Operating Voltage Range:** 2.0 V to 3.6 V

---

#### Calendar Ageing Test Conditions

- **Storage State:** Constant **100% State of Charge (SoC)**
- **Temperature Matrix:** **15°C, 25°C, 35°C, and 45°C**
- **Time Span:** Day 0 to approximately Day 235 (sparse observation checkpoints)

The original study reports **capacity loss** over time. For the iPINN implementation, the data was converted to **remaining capacity**, which serves as the target variable for the neural network:

```text
Q = 1 − Loss
```

where:

- **Q** = Remaining normalized capacity
- **Loss** = Measured capacity fade

Using remaining capacity instead of capacity loss allows the network to directly model battery **State-of-Health (SoH)** evolution while simultaneously enforcing the underlying degradation physics through the PINN framework.

### Experimental Data: Impact of Temperature on Calendar Ageing

![Experimental Data Impact of Temperature on Calendar Ageing](/assets/images/Experimental_Data.jpg)

---

#### Source

The experimental data was extracted from:

> Naumann, M. et al., *Analysis and Modeling of Calendar Aging of a Commercial LiFePO₄/Graphite Cell*, Journal of The Electrochemical Society.

**Paper:**  
https://iopscience.iop.org/article/10.1149/2.1181714jes


### Physics Model

The degradation of lithium-ion batteries is a complex, multidimensional phenomenon involving several interacting ageing mechanisms. While numerous degradation pathways exist, calendar ageing is primarily driven by **SEI (Solid Electrolyte Interphase) growth** and other parasitic side reactions occurring at the electrode-electrolyte interface.

The objective of this work is not to develop a comprehensive battery degradation model, but rather to demonstrate how an **Inverse Physics-Informed Neural Network (iPINN)** can identify physically meaningful degradation parameters directly from sparse experimental observations.

To keep the physics simple and interpretable, we assume that battery capacity loss arises from two mechanisms:

- Diffusion-limited SEI growth
- Long-term parasitic ageing reactions

The derivation of the governing equation is presented below.

---

#### Step 1: Capacity-Loss Model

A commonly used semi-empirical representation of calendar ageing assumes that capacity loss can be approximated by a diffusion-limited t\sqrt{t} term combined with a linear ageing term.

$$ Loss(t) = a\sqrt{t} + bt $$

where:
* $$a$$ = degradation parameter due to SEI growth
* $$b$$ = represents an aggregate long-term ageing contribution arising from slow parasitic reactions not captured by the diffusion-limited SEI term
* $$t$$ = storage time
  
---

#### Step 2: Define Remaining Capacity

The iPINN is trained on **normalized remaining capacity** rather than capacity loss.

$$ Q(t) = 1 - Loss(t) $$

Substituting the capacity-loss model:

$$ Q(t) = 1 - a\sqrt{t} - bt $$
The $$\sqrt{t}$$ term captures the rapid initial degradation associated with SEI growth, while the linear term ($$bt$$) represents slower, continuous ageing mechanisms.


---
#### Step 3: Differentiate with Respect to Time

PINNs enforce governing differential equations, so we require the rate of change of capacity.

Differentiating with respect to time ($$t$$):

$$ \frac{dQ}{dt} = -\frac{d}{dt} \left( a\sqrt{t} + bt \right) $$

Since 

$$ \frac{d}{dt}\left(\sqrt{t}\right) = \frac{1}{2}t^{-0.5} $$

we obtain

$$ \frac{dQ}{dt} = -\left( 0.5a\,t^{-0.5} + b \right) $$

---

#### Step 4: Rearranging into a Governing Equation

Defining

$$ k = 0.5a $$

gives

$$ \frac{dQ}{dt} + k\,t^{-0.5} + b = 0 $$

This differential equation will be used as the governing degradation law for the PINN.

---

#### Step 5: Physical Interpretation

The governing equation contains two physically meaningful degradation mechanisms:

$$ \frac{dQ}{dt} + \underbrace{k\,t^{-0.5}}_{\text{Diffusion-Limited SEI Growth}} + \underbrace{b}_{\text{Long-Term Linear Ageing}} = 0 $$

where:
* $$k\,t^{-0.5}$$ represents degradation due to diffusion-controlled SEI growth.
* $$b$$ represents an aggregate long-term ageing contribution arising from slow parasitic reactions not captured by the diffusion-limited SEI term.
* $$\frac{dQ}{dt}$$ represents the instantaneous capacity-fade rate.

The model naturally predicts that SEI-driven degradation slows over time because the term $$t^{-0.5}$$ decreases as storage time increases.

---

#### Practical Note: Why PINNs Use the Differential Equation

Rather than enforcing the integrated solution

$$ Loss(t) = a\sqrt{t} + bt $$

PINNs enforce the governing differential equation directly.

The physics residual is defined as

$$ R(t) = \frac{dQ_{PINN}}{dt} + k\,t^{-0.5} + b $$

and the corresponding physics loss is

$$ L_{Physics} = \mathrm{MSE}\left( R(t) \right) $$

By minimizing this residual, the PINN learns a solution that not only matches the experimental data but also obeys the governing degradation physics throughout the entire time domain.

The iPINN therefore learns:
* The degradation trajectory $$Q(t)$$
* The SEI-growth parameter $$k$$
* The linear-ageing parameter $$b$$

directly from sparse experimental observations.

---
#### Step 6: Why PINNs Use the Differential Equation

#### **PINN Loss Function Formulation (The most important to understand)**

The overall loss formulation should answer three key questions:
* Does the model match the measured capacity fade?
* Does the model obey the governing physics?
* Does the model start from the correct battery state?

---

#### Physics Loss 

Using the singularity-free formulation, the residual becomes

$$ R(t) = \sqrt{t}\frac{dQ_{PINN}}{dt} + k + b\sqrt{t} $$

The physics loss is then

$$ L_{\text{Physics}} = \mathrm{MSE}\left( R(t) \right) $$

$$ L_{\text{Physics}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| R(t_i) \right|^2 $$

which enforces the governing degradation physics.


>  Practical Note: Singularity Issue and Reformulation

The original governing equation contains the term $$t^{-0.5}$$, which becomes singular near $$t=0$$:

$$ \frac{dQ}{dt} + k\,t^{-0.5} + b = 0 $$

To improve numerical stability, the equation is reformulated by multiplying through by $$\sqrt{t}$$:

$$ \sqrt{t}\frac{dQ}{dt} + k + b\sqrt{t} = 0 $$

The reformulated equation preserves the underlying physics while removing the singularity at the beginning of the time domain.


---

#### Data Loss

The data loss evaluates model fidelity against the sparse measured State-of-Health (SoH) checkpoints:

$$ L_{\text{Data}} = \frac{1}{N_d} \sum_{j=1}^{N_d} \left( Q_{\text{PINN}}(t_j) - Q_{\text{Data}}(t_j) \right)^2 $$

---

#### Initial-Condition Loss

The battery starts at full normalized capacity:

$$ Q(0) = 1 $$

The corresponding initial-condition loss is:

$$  L_{\text{IC}} = \left( Q_{\text{PINN}}(0) - 1.0 \right)^2 $$

which anchors the solution at the correct initial battery state.

---

#### Total Multi-Objective Loss Function

The complete loss function balances data fitting, differential equation compliance, and initial conditions using weighting factors ($\lambda$):

$$ \Large L_{\text{Total}} = \lambda_{\text{Data}} L_{\text{Data}} + \lambda_{\text{Physics}} L_{\text{Physics}} + \lambda_{\text{IC}} L_{\text{IC}} $$

* $$\lambda_{\text{Physics}}$$ balances the trade-off between strict physics adherence and data fitting.
* $$\lambda_{\text{IC}}$$ ensures the trajectory is strongly tied to $Q(0) = 1.0$.

> **Implementation Note:** Setting $\lambda_{\text{Data}} = 1.0$, $\lambda_{\text{Physics}} = 0.1$, and $\lambda_{\text{IC}} = 10.0$ often provides robust initial optimization convergence, mitigating gradient pathologies between data and physics losses.

Now, this formulation ensures that the network simultaneously:
* Fits the measured battery-ageing data,
* Satisfies the governing degradation physics,
* Starts from the correct initial condition,

while discovering physically meaningful degradation parameters directly from sparse experimental observations.

---

#### Practical Note: Enforcing Parameter Positivity Constraints

In an **inverse PINN**, parameters $k$ and $b$ are learned dynamically via backpropagation alongside the network weights. Because kinetic rates cannot be negative ($k, b \ge 0$), we parameterize them using softplus or exponential transformations to guarantee physical plausibility:

$$ \Large k = \text{softplus}(\hat{k}), \quad b = \text{softplus}(\hat{b}) $$

This prevents non-physical predictions where capacity would spontaneously increase over time.

---

