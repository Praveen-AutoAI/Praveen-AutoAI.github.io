---
layout: post
title: "Engineering Application of PINN -  Three-way Catalyst Modelling for Accelerated Product Development"
description: "Automotive Emission modelling using PINN"
date: 2026-08-04
categories: [Machine Learning, Engineering, Scientific Machine Learning, PINN]
tags: [PINN, Physics, Deep Learning, AI]
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


### Three-way Catalytic Converter Simplified Reactions

#### a) CO Oxidation
* **Reaction 1:** $\text{CO} + \frac{1}{2}\text{O}_2 \rightarrow \text{CO}_2$ | 
#### b) HC Oxidation
* **Reaction 2:** $\text{THC} + \text{O}_2 \rightarrow \text{CO}_2 + \text{H}_2\text{O}$ |
#### c) NOx Reduction
* **Reaction 3:** $2\text{CO} + 2\text{NO} \rightarrow 2\text{CO}_2 + \text{N}_2$ | 
* **Reaction 4:** $\text{NO} + \frac{1}{2}\text{O}_2 \leftrightarrow \text{NO}_2$ |
#### d) Ceria Reactions (Oxygen Storage Capacity)
* **Reaction 5:** $2\text{CeO}_2 + \text{CO} \rightarrow \text{Ce}_2\text{O}_3 + \text{CO}_2$ | 
* **Reaction 6:** $\text{THC} + \text{CeO}_2 \rightarrow \text{Ce}_2\text{O}_3 + \text{CO}_2 + \text{H}_2\text{O}$ |
* **Reaction 7:** $\text{Ce}_2\text{O}_3 + \frac{1}{2}\text{O}_2 \rightarrow 2\text{CeO}_2$ | 
* **Reaction 8:** $\text{Ce}_2\text{O}_3 + \text{NO} \rightarrow 2\text{CeO}_2 + \frac{1}{2}\text{N}_2$ | 
#### e) Steam Reforming and Water-Gas Shift Reaction
* **Reaction 9:** $\text{CO} + \text{H}_2\text{O} \rightarrow \text{CO}_2 + \text{H}_2$
* **Reaction 10:** $\text{HC} + \text{H}_2\text{O} \rightarrow \text{CO} + \text{H}_2$
#### f) Hydrogen Oxidation 
* **Reaction 11:** $\text{H}_2 + \frac{1}{2}\text{O}_2 \rightarrow \text{H}_2\text{O}$ | 

For sufficient accuracy in modelling and to have enough degree of freedom to calibrate against the test data, It is a practice to use a more comprehensive set of reactions(I have ued 16 reactions).

### Kinetic Parameters for Calibration: Reaction 1 (CO Oxidation)

To model the chemical kinetics of **Reaction 1**, rate expression is standard. This accounts for both the Arrhenius temperature dependence and the competitive adsorption (inhibition) of species on the active catalyst sites (Pt/Pd). 


The reaction rate of a catalyst reaction is typically represented using an Arrhenius-based expression:

$$
R = \frac{A \exp\left(-\frac{E_a}{R_u T_s}\right) f(y_i)}{G(T_s,y_i)}
$$

where the reaction rate is governed by the catalyst temperature, species concentrations, and a set of kinetic parameters that define the reaction characteristics.

For each reaction, the primary calibration parameters are:

- **Pre-exponential Factor ($A$):** Determines the magnitude of the reaction rate and reflects the frequency of effective molecular interactions.
- **Activation Energy ($E_a$):** Represents the energy barrier that must be overcome for the reaction to occur.

As a result, every reaction introduces at least two unknown kinetic parameters that must be identified through calibration. In a typical three-way catalyst model containing 15 or more reactions, the total number of calibration parameters can become significant. The strong nonlinear coupling between reactions, temperature dynamics, oxygen storage behavior, and species concentrations makes the calibration process highly complex and computationally intensive. Consequently, achieving a parameter set that accurately reproduces experimental data often requires extensive testing, iterative optimization, and considerable domain expertise.


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

For catalyst reaction networks, each reaction introduces multiple kinetic parameters that must be calibrated from experimental data. As the number of reactions increases, the parameter space grows rapidly, making calibration a high-dimensional optimization problem that requires significant computational effort and domain expertise.
