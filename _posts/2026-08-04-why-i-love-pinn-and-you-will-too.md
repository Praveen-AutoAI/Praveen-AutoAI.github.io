---
layout: post
title: "Why I Love PINNs & You Will T2o0 :)"
description: "Why PINN is far better than Deep Learning for Engineers"
date: 2026-08-04
categories: [Machine Learning, Engineering, Scientific Machine Learning, PINN]
tags: [PINN, Physics, Deep Learning, AI]
---

### Why I Love PINNs & You will T20oo :)

The datasets I typically encounter in engineering projects fall into two categories:

#### Case #1: Lots of Data, Lots of Noise

The dataset is large, but the noise is so significant that even the testing engineer starts questioning the measurements.

#### Case #2: Very Little Data, But Credible

The dataset consists of barely 10 data points, but the testing engineer is confident about every one of them. Of course, some noise is always part of engineering measurements, but the data itself is considered trustworthy.

If you have worked in engineering long enough, you know this is not an exception. This is often the norm.

---

### Engineering's Blessing and Curse

Engineering domains have a unique advantage.

Great scientists and engineers had dedicated their lives and **blessed us in the form of governing equations**:

- Heat transfer equations
- Diffusion equations
- Conservation laws
- Electromagnetic equations
- Electrochemical models
- Fluid dynamics equations

These equations describe how systems should behave.

At the same time, engineering comes with **Curse of uncertainty***:

- Component-to-component variation
- Manufacturing tolerances
- Process variation
- Measurement uncertainty
- Sensor inaccuracies
- Test-to-test variation

So while physics is often well understood, the data is rarely perfect.

---

### My Deep Learning Dilemma

I have always been a fan of deep learning.

Partly because I am lazy 🙂

The workflow is attractive:

```text
Load Data
    ↓
Train Neural Network
    ↓
Find Function
```

Simple.

Unfortunately, real-world engineering projects introduced two massive challenges.

#### Challenge #1: Sparse Data Availability

Most of the time. You would get 15-20 data points and they call it a dataset ;(

---

#### Challenge #2: Convincing Engineering Domain Experts 

Always the challenge. Even after building a highly accurate model, the first question during reviews is usually:

> "Why should I trust this prediction?"

And honestly, that is a perfectly reasonable question. Engineering experts trust physics more than black-box models. I should admit that those critical questions from them was the drive for me to learn and apply PINN at my work.

---

### Can Machine Learning be used as a Scientific tool? How to make it "Scientific"?

### Enter Physics-Informed Neural Networks (PINNs)

The savior was **Physics-Informed Neural Networks (PINNs)**.

PINNs helped address both problems simultaneously.

Instead of training a neural network purely on data, PINNs incorporate known governing equations directly into the loss function during training.

In simple terms, the model is forced to satisfy:

1. The observed data
2. The known physics

at the same time.

The loss function becomes:

```text
Loss = Loss_Data + λ × Loss_Physics
```

where:

- **Data Loss** ensures agreement with measurements.
- **Physics Loss** ensures compliance with governing equations.

### PINN Architecture
![PINN Architecture](/assets/images/PINN_Intro.png)

---

### Why This Is Beautiful

With PINNs, even when data is scarce, the model is guided by decades of scientific knowledge embedded in the governing equations.

This dramatically improves generalization and most importantly interpretability (Able to explain the model's prediction is considered so important in engineering domains).

---

### Solving the Reviewer Problem

One interesting impact of PINNs is how they change technical discussions.

With a conventional neural network, the conversation often becomes:

> "How do we know this model will extrapolate correctly?"
> How to explain the prediction, What is the inference from the prediction?"
> "I have a legacy mathematical model why should i trust this? "

With a PINN, the conversation shifts to:

> "The model is fitting the data while obeying the same governing equations used in our engineering analysis."
> "The governing equation are these and coefficient derived conveys such a trend"

That creates a completely different **level of trust**.

---

### What Excites Me Most

The most exciting aspect of PINNs is that they are not just prediction tools.

They can become **scientific discovery tools**.

Instead of finding only outputs, they can infer hidden physical parameters such as:

- Diffusion coefficients
- Reaction rate constants
- Material properties
- Thermal constants
- Battery degradation rates

This moves the problem from pure prediction:
> "What will happen?"
to propogation into the prediction 
> "What physical mechanism is causing this behavior?"
And that is incredibly powerful.

### Challenges in PINN:
- Selecting the governing equation - Adding all the governing equation can lead to fiasco => adding the right, simple and appropriate enough equation will lead to successful deployment
- Training instability & lack of convergence - Sometimes you feel you are in wild forest without any clue => progressive addition of physics equation, adjusting the learning rate for weights and physics parameters are needed to solve.
- Computation intensive - Especially for higher-order derivative and complex multi-physics system => better to approximate to first-order system 

---

### Final Thoughts

Advantages of PINN:
- Works well with sparse datasets 
- Physics-Based generalization (using our existing knowledge in the form of mathematical governing laws)
- Higher trust and explainability
- Reduced dependence on noisy data (noise from sources like sensor noise, measurement uncertainty, manufacturing variations, etc.)
- Enables parameter discover (Intering meaning physical parameters & this area if PINN is called as Inverse PINN)
- Build models that engineers can actually believe (Amazing for Engineering AI)

I still love deep learning. But engineering problems are fundamentally different from image classification or language modeling problems. We rarely have unlimited data. What we do have is physics. Physics-Informed Neural Networks provide a natural way to combine both worlds.

Most important advantage in my experience are, they help in encountering both:

- the **noisy and sparse characteristics of engineering data**, and
- the **nosy and critical characteristics of engineering reviewers** 🙂

### And that is why I love PINNs and you will t2o.

### And that is why PINN is given the name "Scientific" Machine Learning (SciML). 
### **Though not transparent it's as translucent Glass-Box and not a black-box anymore!!!**

### <span style="color:#3b82f6;">Though not transparent it's as translucent Glass-Box and not a black-box anymore!!!</span>

![Scientific_Machine_Learning](/assets/images/PINN/SciML.png)

### Gentle References

1. [MathWorks: Physics-Informed Neural Networks](https://in.mathworks.com/discovery/physics-informed-neural-networks.html)
2. [A Short Introduction to Physics-Informed Neural Networks (PINNs)](https://medium.com/@vivek-karmarkar/a-short-introduction-to-physics-informed-neural-networks-pinns-cd342f5a3c5e)
3. [So What Is a Physics-Informed Neural Network?](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
4. [PINNs Introductory Code for the Heat Equation](https://dcn.nat.fau.eu/pinns-introductory-code-for-the-heat-equation/)
5. [PINNs-TF2.x GitHub Repository](https://github.com/Vaezi92/PINNs-TF2.x)
6. [Steve_Brunton_PINN_Intro (cool video series)](https://www.youtube.com/watch?v=JoFW2uSd3Uo&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa)
