---
layout: post
title: "Lessons Learned(& Learning) While Building Physics-Informed Neural Networks (PINNs)"
description: "Practical challenges and solutions when developing PINNs for engineering applications."
date: 2026-08-06
categories: [Machine Learning, Engineering, Physics]
tags: [PINN, Deep Learning, Scientific Computing, Engineering AI]
math: true
---

## Solving the Mysteries of PINN training

One of the reasons I(being an Automotive Engineer turned AUTO+AI guy) enjoy working with Physics-Informed Neural Networks (PINNs) is that they force me to think like both a machine learning engineer and a physicist.

Unlike conventional neural networks that primarily focus on minimizing a data-driven loss, PINNs must simultaneously satisfy experimental observations and governing physical laws. This additional constraint is precisely what makes them powerful, but it is also that makes them vulnerable and challenging to train.

During my journey building PINNS & Inverse PINNs in my projects, I ran into several practical challenges. Some were numerical, some were physical, and some were simply optimization problems disguised as physics problems, but all were like mysterious issues at first. 
**This post could save you from lot of mysterious problems that you would encounter during your first PINN project.**

I have listed the common bottlenecks while handling PINN projects.(most of these I have faced and I wanted to put them as lessons learned)

---

### 1. Selecting the Right Physics/Governing laws/Boundary and Initial Conditions
**One Line: High Physics fidelity does not promise good PINN model**

Selecting the right governing physics equations is the foundational bottleneck in PINN design. Because a PINN uses physical laws as a regularizer in the loss function, embedding equations that under-represent the physical system or over-complicate it creates an immediate conflict during training. 
Just because the governing laws are available(comprehensive but complex due to high-dimension PDE, Mutli-physics), it is not the right practice to implement the same as physics constraint. 
This presents a "Goldilocks problem" where the equation set must be sophisticated enough to capture dominant system behaviors, but simple enough to avoid numerical stiffness during automatic differentiation.

#### Best Practices:
- Always start with one fundamental law of the system(still keep it simple, check the Calendar ageing PINN demo link at the bottom), and try put additional constraint with multiple equations, boundary and initial conditions (**Follow the "Occam's Razor Rule"**)
- If you got high dimensional PDE, convert to first order PDE for simplicity. This will also help in faster training.
- Non-dimensionalize variables using characteristic length and time values
- Add boundary & Initial conditions constraint that will support the learning of the system characteristics. Avoid non-essential stuffs.

**Idea_of_PINN**
![Idea_of_PINN/Importance](/assets/images/PINN/PIML.jpg)

Remember that even with some appropriate or partially known physics, PINNs will perform (more precisely, you can enable PINNs to perform).

---

### 2. Noise in the Data (in case of Inverse PINN)
**One Line: Large noise may lead to nasty PINN**

When using real-world sensor/experimental data for inverse problems, noisy observations degrade parameter accuracy and can mislead the physics residual calculations. If the noise is within a threshold of less than 2%, as a matter of fact it helps in model generalization, so check the data credibility. (I always add a controlled noise for the generalization effect)

#### Best Practices:
- Understand the source of data, check the spread and variability.
- If you got the plant model (engine/vehicle performance model/ battery P2D model, etc.) a good way would be to add simulated data in order to reduce the dependency on the experimental data.

---

### 3. Handling Mathematical Singularities
One Line: Blind implementation of the physics laws won't give the vision to the PINN model
Neural networks and automatic differentiation do not behave well when the governing equation contains singular terms.

Expressions such as:

$$
\frac{1}{t}, \qquad \frac{1}{\sqrt{t}}, \qquad t^{-1}, \qquad t^{-0.5}
$$

can generate extremely large residuals near zero.

The result is often:
- Unstable gradients ; Parameter collapse ; NaN values ; Failed training runs

NaN are serious problem in PINN training, monitoring the training loss of all the loss components (it will help you to diagnose PINN)

#### Best Practices:
- Whenever possible, reformulate the governing equation.
For example:

$$
\frac{dQ}{dt} + \frac{k}{\sqrt{t}} = 0
\;\Longrightarrow\;
\sqrt{t}\,\frac{dQ}{dt} + k = 0
$$


Both equations represent the same physics, but the second form is significantly more stable for neural network optimization. I suggest you to verify your physics constraints for singularities and adapt them.
For my battery-aging PINN demo project on my GitHub repo, this reformulation was one of the most important improvements I made.

---

### 4. Gradient Pathology / Loss Balancing
**One Line: Find the right balance between the loss components to establish peace during trianing.**

Total Loss function:

$$
L_{total}=\lambda_{data}L_{data}+\lambda_{phys}L_{phys}+\lambda_{IC}L_{IC}
$$

The problem occurs when one loss produces much larger gradients than the others.
For example:
Data loss gradients dominate → model fits experimental data well but violates physics.
Physics loss gradients dominate → model satisfies equations but poorly fits data.
Loss terms continuously fight each other → unstable convergence or oscillations.

#### Best Practices:
- If you are lucky manually weighing lambdas would work ;) , Check the scale of loss while training and you can methodically adjust the weights.
- Implementing dynamic loss weighting using GradNorm, self-adaptive weights is a wise option if you got really a complex and competing loss components

---

### 5. Enforcing Physical Constraints on Parameters
**One Line: Do the reality check for sign and magnitude of parameters**

Neural networks are excellent at function approximation. They are not physicists. PINN may find mathematically convenient but physically impossible parameters if those values happen to reduce the loss function. For example
- negative degradation rates, negative mass
- Out of range
These are not useful simply because you cannot interpret physically.

#### Best Practices:
- Embed physical constraints directly into the model. 
For positive-only parameters, transformations such as:
```python
torch.abs(parameter)
```
or preferably:
```python
torch.nn.functional.softplus(parameter)
```
- Let's we want a parameter to in the range of 0 to 1. Instead of training k directly, train an unconstrained variable θ\thetaθ and map it using a sigmoid function (squeezes the output between 0 to 1).
```python
theta = nn.Parameter(torch.randn(1))
k = torch.sigmoid(theta)
```

For a parameter $p$ constrained to lie within the range $[p_{\min},\, p_{\max}]$, an unconstrained optimization variable
\(\theta\) can be transformed as

$$
p = p_{\min} + \left(p_{\max} - p_{\min}\right)\sigma(\theta)
$$

where

$$
\sigma(\theta) = \frac{1}{1 + e^{-\theta}}
$$

is the sigmoid function. This guarantees that

$$
p_{\min} \leq p \leq p_{\max}
$$

throughout training.

---

### 6. Collocation Point Distribution Matters
**One Line: Sparse collocation points leads to a porous PINN model**

Physics is only enforced where the residual is evaluated. If points are too sparse or evenly distributed, the network might satisfy the equations at those specific dots but behave wildly elsewhere.

#### Best Practices:
- Use sufficient number of collocation points across the domain, specifically clustering them in regions with rapid physical changes or steep gradients (like the initial period lithium-ion cell ageing since the degradation is rapid)
- There is no standard to choose the number of Collocation points, but typical I choose 25 times the experimental data points.
- And the number of collocation points should be increased incase of higher order PDEs

$$
\begin{array}{|l|c|}
\hline
\textbf{Problem\ Type} & \textbf{Typical\ Collocation\ Points} \\
\hline
\text{Simple ODE} & 10^{2} - 10^{3} \\
\hline
\text{Coupled ODEs} & 10^{3} - 10^{4} \\
\hline
\text{1D PDE} & 10^{4} - 10^{5} \\
\hline
\text{2D PDE} & >10^{5} \\
\hline
\text{Sharp Fronts / Discontinuities} & \text{Adaptive Sampling Required} \\
\hline
\end{array}
$$

> A PINN learns physics only where you ask it to enforce physics.

---
### 7. Optimization Plateaus
**One Line: You need robust strategy to find the global minima of the majestic and wild Himalayan(loss landscape) range**
The PINN loss function combines multiple objectives (data fitting, physics equations, boundary conditions), creating a highly complex and non-convex optimization landscape with many flat regions, saddle points, and local minima. Optimizers such as Adam rapidly reduce the loss during the initial stages of training but often stagnate once they reach a plateau. As a result, the model may appear converged while significant physics residuals still remain, leading to solutions that satisfy the data reasonably well but lack the accuracy required for scientific and engineering applications.

#### Best Practices:

- A common PINN training strategy is:
1. Train with **Adam** for rapid initial exploration. (Glides through the slope quickly)
2. Switch to **L-BFGS** for fine tuning. (utilizes curvature information to make more informed parameter updates)

L-BFGS is particularly effective at reducing the physics residual to very small values and is used extensively in many successful PINN implementations.

---

### 8. Learn to Read the Loss Components, Not Just the Total Loss
**One Line: Shape of the loss curves can deceive you, Scale can assert you**
One of the biggest mistakes I made early on was assuming that a noisy loss curve automatically meant that something was wrong with the training.

PINNs optimize multiple objectives simultaneously:

- Data loss
- Physics loss
- Initial-condition loss
- Boundary-condition loss (if present)

As a result, it is perfectly normal for one loss component (especially which has the smallest magnitude) to oscillate while the others converge smoothly.

Check **my GitHub project repo**: For example, during the Calendar Ageing iPINN  project, the Initial Condition (IC) loss exhibited noticeable fluctuations throughout training. At first glance, it appeared unstable. However, a deeper investigation revealed that:

- The magnitude of the IC loss was extremely small (on the order of **10⁻⁷**).
- The predicted initial capacity remained essentially equal to 1 (ie. 0.9999).
- The total loss, data loss, and physics loss had already converged.

In other words, the network was behaving correctly despite the oscillations.

#### Best Practices:
- Do not judge PINN training solely by shape of the curves. Examine the magnitude of each loss component.
- Check whether the associated physical constraint is actually being satisfied.
- Focus on whether the predicted physical fields and parameter values make sense.
- Verify that the model behaviour remains physically meaningful.

> In PINNs, loss curves are diagnostic tools, not pass/fail indicators. Always interpret the meaning behind a loss component before concluding that the training is unstable.

### Final Thoughts

The hardest part of building a PINN is rarely the neural network.

The real challenge lies in converting physical knowledge into a numerically stable optimization problem.

In my experience, most PINN failures can be traced back to one of the causes:

- Too complicated physics constraints
- Poor loss balancing
- Mathematical singularities
- Missing physical constraints
- Optimizer limitations
- Unstable training and poor convergence
- Inadequate collocation-point selection (not often, but for higher order PDEs)

Once these challenges are addressed, PINNs become incredibly powerful tools for combining sparse experimental data with known physics and discovering hidden system parameters.

And perhaps that is what makes PINNs so fascinating.

They are not just machine learning models.

They are a conversation between data and physics.

### References

[1] Emergent Mind, *Inverse Physics-Informed Neural Networks (Inverse PINNs).*  
Available: [https://www.emergentmind.com/topics/inverse-pinn](https://www.emergentmind.com/topics/inverse-pinn)

[2] Joris C., *Physics-Informed Machine Learning (PIML),* Medium.  
Available: [https://medium.com/@joris.c/physics-informed-machine-learning-piml-debe8f856c10](https://medium.com/@joris.c/physics-informed-machine-learning-piml-debe8f856c10)

[3] P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell,  
*"Challenges in Training PINNs: A Loss Landscape Perspective,"*  
arXiv:2402.01868, 2024.  
Available: [https://arxiv.org/abs/2402.01868](https://arxiv.org/abs/2402.01868)

[4] S. M. Abbas, M. S. A. Ahamed, M. E. H. Chowdhury, et al.,  
*"Review of Physics-Informed Neural Networks: Challenges in Loss Function Design and Geometric Integration,"*  
Mathematics, vol. 13, no. 20, 3289, 2025.  
Available: [https://www.mdpi.com/2227-7390/13/20/3289](https://www.mdpi.com/2227-7390/13/20/3289)

[5] Y. Wang, J. Fan, S. Wang, and X. Liu,  
*"Solving Real-World Optimization Tasks Using Physics-Informed Neural Computing,"*  
Scientific Reports, Nature Portfolio, 2023.  
Available: [https://www.nature.com/articles/s41598-023-49977-3](https://www.nature.com/articles/s41598-023-49977-3)
