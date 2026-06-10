# Linearized subspace refinement framework to expose hidden accuracy in trained neural networks

**Wenbo Cao**, **Weiwei Zhang***  
a School of Aeronautics, Northwestern Polytechnical University, Xi’an 710072, China  
b International Joint Institute of Artificial Intelligence on Fluid Mechanics, Northwestern Polytechnical University, Xi’an 710072, China  
c National Key Laboratory of Aircraft Configuration Design, Xi’an 710072, China  

---

## Abstract

Neural networks trained by gradient-based methods often exhibit optimization-induced accuracy plateaus in scientific machine learning tasks. We present Linearized Subspace Refinement (LSR), an architecture-agnostic post-training framework that exploits the local linearized model at a fixed trained state. By solving a reduced direct least-squares problem in a Jacobian-defined low-dimensional space, LSR computes a subspace-optimal linearized correction and yields a refined predictor with markedly improved accuracy. Across function approximation, data-driven operator learning, physics-informed operator fine-tuning, and noisy inverse problems, LSR shows that standard nonlinear training can remain far above this subspace-attainable error level. Similar accuracy plateaus persist even for the convex quadratic problem from local linearization when solved with standard iterative optimizers, identifying numerical ill-conditioning as a primary bottleneck. LSR frequently delivers order-of-magnitude error reductions, while the subspace rank provides an explicit capacity-control mechanism that balances correction strength, numerical stability, and noise sensitivity. Together, LSR exposes conditioning-limited attainable accuracy in trained-state linearized models and provides direct access to it. 

---

## Data-Driven Operator Learning

The data-driven operator learning experiments and baseline implementations are adapted from:

> https://github.com/yaohua32/Deep-Neural-Operators-for-PDEs/tree/main

Please follow the **original repository README** to correctly download and prepare the required datasets before running the corresponding scripts in this project.  

---

## Physics-Informed Operator Learning

The physics-informed operator learning experiments are based on the implementation provided by:

> https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

- The original training code in that repository is implemented in **JAX**.
- In this project, the trained JAX models are converted to **PyTorch**.
- A PyTorch-based implementation of **LSR** is then applied on top of the converted models.

### Burgers Equation in physics-informed operator learning 

For the Burgers equation experiments, the training data must be generated **following the instructions in the original repository README**.  
Please ensure that the data generation step is completed before running the corresponding scripts in this project.

---

## Additional Experiments

- **Function approximation:**     All scripts for reproducing the function-approximation experiments, including model training and LSR evaluation, are included in this repository. The datasets used in these experiments are generated directly by the provided scripts.  
- **Rank-controlled LSR for noisy inverse problems:**     All scripts for reproducing the rank-controlled noisy inverse-problem experiments are included in this repository. Required configuration files and small auxiliary data files are provided alongside the code. 
