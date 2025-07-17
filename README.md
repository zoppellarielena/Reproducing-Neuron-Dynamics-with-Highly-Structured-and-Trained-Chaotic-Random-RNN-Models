# 🧠 Discrete Attractor Dynamics in the Frontal Cortex  
## Reproducing Neuron Dynamics with Highly Structured and Trained Chaotic Random RNN Models

This project explores how persistent neural activity, firing that continues after a stimulus is removed, can be reproduced through both **biophysically structured models** and **trained chaotic recurrent neural networks (RNNs)**.

Persistent activity is crucial for short-term memory and motor planning. In their study, [Inagaki et al. (2019)](https://doi.org/10.1038/s41586-019-0919-7) demonstrated that such activity in the **anterior lateral motor cortex (ALM)** of mice arises from **network dynamics**, not intrinsic cellular properties. Their results were modeled using structured discrete attractor networks.

This project asks:  
**What happens if we instead train a chaotic random RNN to reproduce the same dynamics observed during a delayed paired-association task?**


## Inagaki et al. Experimental Background

- **Task:** Delayed-response left/right licking task
- **Epochs:** Sample → Delay → Go cue
- **Data:** Neural recordings from ALM using silicon probes
- **Key feature:** Persistent activity observed during the delay epoch

## Project Goals

This project is divided into two main parts:

### **1. Reproducing Inagaki et al. (2019)**

- Analyze extracellular recordings from the ALM replicating key analyses:
  - **Coding Direction (CD)** projection
  - Neural variability over time
  - Photoinhibition response
- Simulate the **three-population structured attractor model** (Excitatory Left, Excitatory Right, Inhibitory)
- Show that fixed parameters in the structured model can reproduce persistent activity and discrete attractor states

### **2. Training a Chaotic Random RNN via FORCE Learning**

Following the method of [Rajan et al. (2016)](https://doi.org/10.1016/j.neuron.2016.02.009), we:

- Implement a chaotic RNN where **all synapses are trainable** using the FORCE algorithm
- Design task-aligned external input:
  - Identical signals for both left/right trials during **pre-sample** and **delay**
  - Divergent signals only during **sample** and **after-cue** epochs
- Train on trial-averaged activity from **87 neurons** (largest available dataset)
- Also train a variant with **explicit suppression of non-selective activity**
- Evaluate performance using:
  - **CD projection**
  - **Selectivity measures**
  - **Robustness tests** (e.g., noise perturbation, photoinhibition)
- Analyze network structure:
  - Synaptic weight distributions
  - **PCA** and **eigenvalue spectrum** of connectivity matrices

---

## Main Findings

- **Structured attractor models** successfully replicate persistent activity and discrete state transitions under photoinhibition
- **Random RNNs** trained with FORCE can **memorize left/right trial inputs** during the delay period
- **Noise perturbations** during the delay can cause **right-trial trajectories to collapse into left-trial dynamics**

