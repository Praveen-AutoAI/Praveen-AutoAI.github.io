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



---

### Source

The experimental data was extracted from:

> Naumann, M. et al., *Analysis and Modeling of Calendar Aging of a Commercial LiFePO₄/Graphite Cell*, Journal of The Electrochemical Society.

📄 **Paper:**  
https://iopscience.iop.org/article/10.1149/2.1181714jes

