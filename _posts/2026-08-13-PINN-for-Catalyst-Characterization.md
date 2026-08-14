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

The reaction rate $R_1$ is typically expressed as:

$$R_1 = \frac{A \exp\left(-\frac{E_a}{R_u T_s}\right) y_{\text{CO}} y_{\text{O}_2}}{G(T_s, y_i)}$$

To calibrate this specific reaction, the following empirical parameters must be identified:

**1. Main Arrhenius Kinetic Parameters**
* **$A$ (Pre-exponential or Frequency Factor):** Represents the collision frequency of the reacting molecules.
* **$E_a$ (Activation Energy):** The minimum energy barrier required to initiate the CO oxidation reaction (typically in $\text{J/mol}$).

For each reaction you will have 2 unknown parameters to calibrate. For a reaction set of 15+ the calibration parameters is so large that it become difficult to calibrate. This actually demands a good domain experience to understand the multi-dimensional effect of the parameters to reach a succesfull parameter set aggreging to the test data.
