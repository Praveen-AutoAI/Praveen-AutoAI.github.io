---
layout: post
title: "Engineering Application of PINN -  Three-way Catalyst Modelling for Accelerated Product Development"
description: "Automotive Emission modelling using PINN"
date: 2026-08-04
categories: [Machine Learning, Engineering, Scientific Machine Learning, PINN]
tags: [PINN, Physics, Deep Learning, AI]
math: true
---

## Background

- **Three-way catalysts (TWCs)** are a critical component of gasoline engine exhaust aftertreatment systems, enabling the simultaneous conversion of harmful carbon monoxide (CO), unburned hydrocarbons (HC), and nitrogen oxides (NOx) into less harmful emissions. The behavior of a **TWC is governed by the coupled effects of exhaust gas transport, heat transfer, oxygen storage and release, and catalytic surface reactions** occurring within the catalyst substrate. 

- A dynamic catalyst models should be capable of capturing the interaction between engine-out emissions, catalyst temperature, and oxygen storage state, enabling the prediction of conversion efficiency under transient operating conditions such as cold start, load changes, and rich-lean transitions.

- In Automotive OEMs, we **use TWC models for ECU control strategy development, model-based calibration, onboard diagnostics (OBD), and virtual vehicle simulations, helping improve calibration efficiency, reduce development and testing effort**, and support the design of emission-compliant powertrain systems.

## Automotive Catalytic Converter
![Catalytic Converter](/assets/images/Emissions/Catalyst.jpg)

## Objective

- Conventional three-way catalyst (TWC) modeling is typically based on **detailed reaction kinetics and distributed parameter formulations, requiring significant calibration effort and domain expertise**. Although commercial tools provide high-fidelity modeling capabilities, the calibration process is often time-intensive, typically requiring 6-8 weeks of dedicated engineering effort.

- A faster and sufficiently accurate TWC modeling approach can significantly **accelerate early-stage powertrain and aftertreatment development** by enabling rapid assessment of emission system performance, reducing dependency on extensive test campaigns, and supporting key design decisions during upstream development.

- The objective of this work is to **develop a data-driven Physics-Informed Neural Network (PINN) based TWC model that combines physical knowledge with experimental data to capture catalyst dynamics efficiently**. The proposed approach aims to support digital vehicle emission simulations, improve calibration efficiency, and reduce the overall development and testing effort while maintaining the level of accuracy required for engineering applications.

## Need for Some Chemistry!!!
For a general chemical reaction,

$$
A + B \rightarrow C
$$

the reaction rate can be expressed as:

$$
R = k \, [A]^m [B]^n
$$

where the rate constant follows the Arrhenius equation:

$$
k = A \exp\left(-\frac{E_a}{RT}\right)
$$

To model the reaction, the key parameters that typically need to be identified are:

- **Pre-exponential Factor ($A$):** Determines the overall magnitude of the reaction rate.
- **Activation Energy ($E_a$):** Defines the temperature sensitivity of the reaction.
- **Reaction Orders ($m$, $n$):** Describe the influence of reactant concentrations on the reaction rate.

<p style="color:blue;">
<strong>Remember This:</strong> For catalyst reaction networks, each reaction introduces multiple kinetic parameters that must be calibrated from experimental data. As the number of reactions increases, the parameter space grows rapidly, making calibration a high-dimensional optimization problem that requires significant computational effort and domain expertise.
</p>

![Catalytic Converter_Reactions](/assets/images/Emissions/Reactions.png)

By the way, this is a simplified reaction set. For sufficient accuracy in modelling and to have enough degree of freedom to calibrate against the test data, It is a **practice to use a more comprehensive set of reactions(I have used 16 reactions)**. 

We know Reaction Orders, but the **Pre-exponential Factor and Activation Energy need be calibrated for each reaction**, summing upto **25+ parameters to calibrate in order to achieve accurate enough model** with target error less than +/-20%. Yeah, Emission system modelling is highly challenging and achieving +/-20% for different inlet conditions is actually amazing.

## 
