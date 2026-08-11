---
layout: post
title: "Lessons Learned(& Learning) While Building Physics-Informed Neural Networks (PINNs)"
date: 2026-08-06
categories: [Machine Learning, Engineering, Physics]
tags: [PINN, Deep Learning, Scientific Computing, Engineering AI]
---

## Lessons Learned While Building Physics-Informed Neural Networks (PINNs)

One of the reasons I(being an Automotive Engineer turned AUTO+AI guy) enjoy working with Physics-Informed Neural Networks (PINNs) is that they force me to think like both a machine learning engineer and a physicist.

Unlike conventional neural networks that primarily focus on minimizing a data-driven loss, PINNs must simultaneously satisfy experimental observations and governing physical laws. This additional constraint is precisely what makes them powerful, but it is also that makes them vulnerable and challenging to train.

During my journey building PINNS & Inverse PINNs in my projects, I ran into several practical challenges. Some were numerical, some were physical, and some were simply optimization problems disguised as physics problems.

I have listed the common bottlenecks while handling PINN projects.(most of these I have faced and I wanted to put them as lessons learned)

---

### 1. Selecting the Right Physics/Governing laws/Boundary and Initial Conditions
One Line: High Physics fidelity does not mean good PINN modelling
Selecting the right governing physics equations is the foundational bottleneck in PINN design. Because a PINN uses physical laws as a regularizer in the loss function, embedding equations that under-represent the physical system or over-complicate it creates an immediate conflict during training. 
Just because the governing laws are available(comprehensive but complex due to high-dimension PDE, Mutli-physics), it is not the right practice to implement the same as physics constraint. 
This presents a "Goldilocks problem" where the equation set must be sophisticated enough to capture dominant system behaviors, but simple enough to avoid numerical stiffness during automatic differentiation.

#### Best Practices:
- Always start with one fundamental law of the system(still keep it simple, check the Calendar ageing PINN demo link at the bottom), and try put additional constraint with multiple equations, boundary and initial conditions (Follow the "Occam's Razor Rule")
- If you got high dimensional PDE, convert to first order PDE for simplicity. This will also help in faster training.
- Add boundary & Initial conditions constraint that will support the learning of the system characteristics. Avoid non-essential stuffs.

---

### 2. Noise in the Data (in case of Inverse PINN)
One Line: Large noise may lead to nasty PINN
When using real-world sensor/experimental data for inverse problems, noisy observations degrade parameter accuracy and can mislead the physics residual calculations. If the noise is within a threshold of less than 2%, as a matter of fact it helps in model generalization, so check the data credibility. (I always add a controlled noise for the generalization effect)

#### Best Practices:
- Understand the source of data, check the spread and variability.
- If you got the plant model (engine/vehicle performance model/ battery P2D model, etc.) a good way would be to add simulated data in order to reduce the dependency on the experimental data.

---

### 3. Handling Mathematical Singularities

Neural networks and automatic differentiation do not behave well when the governing equation contains singular terms.

### The Pitfall

Expressions such as:

```text
1 / t
1 / √t
t⁻¹
t⁻⁰·⁵
```

can generate extremely large residuals near zero.

The result is often:

- Unstable gradients
- Parameter collapse
- NaN values
- Failed training runs

NaN are serious problem in PINN training, monitoring the training loss of all the loss components (it will help you to diagnose PINN)

### The Fix

Whenever possible, reformulate the governing equation.

For example:

```text
dQ/dt + k·t⁻⁰·⁵ = 0
```

can be rewritten as:

```text
√t·dQ/dt + k = 0
```

Both equations represent the same physics, but the second form is significantly more stable for neural network optimization.

For my battery-aging PINN, this reformulation was one of the most important improvements I made.

---

### 4. Gradient Pathology / Loss Balancing
One Line: How the different loss terms are weighted during training.
Total Loss function:
$$
L_total = λ_data L_data + λ_phys L_phys + λ_IC L_IC
$$
The problem occurs when one loss produces much larger gradients than the others

### The Pitfall

The model may:

- Ignore the physics and focus entirely on fitting the data.
- Satisfy the governing equation perfectly while ignoring the measurements.
- Oscillate indefinitely because different loss components pull the model in conflicting directions.

### The Fix

Carefully balance the loss terms using weighting factors such as:

```text
L_total = λ_data L_data + λ_phys L_phys + λ_IC L_IC
```

Finding the right balance between data, physics, and boundary-condition losses is often one of the most important tuning activities in PINN training.

---

### 2. Handling Mathematical Singularities

Neural networks and automatic differentiation do not behave well when the governing equation contains singular terms.

### The Pitfall

Expressions such as:

```text
1 / t
1 / √t
t⁻¹
t⁻⁰·⁵
```

can generate extremely large residuals near zero.

The result is often:

- Unstable gradients
- Parameter collapse
- NaN values
- Failed training runs

NaN are serious problem in PINN training, when monitoring the training loss when you see NaN, you will feel real depressed ;)

### The Fix

Whenever possible, reformulate the governing equation.

For example:

```text
dQ/dt + k·t⁻⁰·⁵ = 0
```

can be rewritten as:

```text
√t·dQ/dt + k = 0
```

Both equations represent the same physics, but the second form is significantly more stable for neural network optimization.

For my battery-aging PINN, this reformulation was one of the most important improvements I made.

---

### 3. Enforcing Physical Constraints on Parameters

Neural networks are excellent optimizers.

They are not physicists.

### The Pitfall

An Inverse PINN may happily discover:

- Negative degradation rates
- Negative diffusivities
- Negative reaction constants
- Other mathematically convenient but physically impossible solutions

if those values happen to reduce the loss function.

### The Fix

Embed physical constraints directly into the model.

For positive-only parameters, transformations such as:

```python
torch.abs(parameter)
```

or preferably:

```python
torch.nn.functional.softplus(parameter)
```

help ensure that the learned parameters remain physically meaningful throughout training.

---

## 4. Optimizer Limitations

The optimization landscape of a PINN is considerably more complex than that of a conventional neural network.

### The Pitfall

Training can plateau early even when the physics residual remains relatively large.

Adam often finds a reasonable solution quickly, but may struggle to achieve the precision required for scientific applications.

### The Fix

A common PINN training strategy is:

1. Train with **Adam** for rapid initial exploration.
2. Switch to **L-BFGS** for fine tuning.

L-BFGS is particularly effective at reducing the physics residual to very small values and is used extensively in many successful PINN implementations.

---

## 5. Collocation Point Distribution Matters

Physics is only enforced where the residual is evaluated.

### The Pitfall

If collocation points are sparse, the network may satisfy the governing equation at those locations while behaving poorly between them.

In other words, the physics is only learned where it is enforced.

### The Fix

Use a sufficiently dense set of collocation points across the domain.

Even better, concentrate additional collocation points in regions where rapid physical changes occur.

For battery-aging problems, this often means placing more collocation points near the beginning of life where degradation occurs more rapidly.

> A PINN learns physics only where you ask it to enforce physics.

The distribution of collocation points is therefore just as important as the experimental measurements themselves.

---
---

### 6. Learn to Read the Loss Components, Not Just the Total Loss

One of the biggest mistakes I made early on was assuming that a noisy loss curve automatically meant that something was wrong with the training.

### The Pitfall

PINNs optimize multiple objectives simultaneously:

- Data loss
- Physics loss
- Initial-condition loss
- Boundary-condition loss (if present)

As a result, it is perfectly normal for one loss component to oscillate while the others converge smoothly.

For example, during the Calendar Ageing iPINN  project, the Initial Condition (IC) loss exhibited noticeable fluctuations throughout training. At first glance, it appeared unstable. However, a deeper investigation revealed that:

- The magnitude of the IC loss was extremely small (on the order of **10⁻⁷**).
- The predicted initial capacity remained essentially equal to 1.
- The total loss, data loss, and physics loss had already converged.

In other words, the network was behaving correctly despite the oscillations.

### The Fix

Do not judge PINN training solely by the appearance of a loss curve.

Instead:

- Examine the magnitude of each loss component.
- Check whether the associated physical constraint is actually being satisfied.
- Look at the learned parameters and predicted solutions.
- Verify that the model behaviour remains physically meaningful.

A noisy loss term is not necessarily a problem. What matters is whether the underlying physical constraint is being violated.

### Key Takeaway

> In PINNs, loss curves are diagnostic tools, not pass/fail indicators. Always interpret the meaning behind a loss component before concluding that the training is unstable.

Sometimes a wildly oscillating loss is simply telling you that the error is already extremely small. Understanding the story behind the loss is often more important than the loss value itself.

## Final Thoughts

The hardest part of building a PINN is rarely the neural network.

The real challenge lies in converting physical knowledge into a numerically stable optimization problem.

In my experience, most PINN failures can be traced back to one of five causes:

- Poor loss balancing
- Mathematical singularities
- Missing physical constraints
- Optimizer limitations
- Inadequate collocation-point selection

Once these challenges are addressed, PINNs become incredibly powerful tools for combining sparse experimental data with known physics and discovering hidden system parameters.

And perhaps that is what makes PINNs so fascinating.

They are not just machine learning models.

They are a conversation between data and physics.
