# Memory Is All You Need

> **Explicit Long-Term Memory as a First-Class Architectural Component**

This project explores a core question that modern deep learning systems largely avoid:

> **What if memory were not an emergent side-effect of attention, but an explicit, controllable, and optimizable subsystem?**

Instead of extending context windows or stacking attention layers, this repository implements and studies **neural architectures with explicit long-term memory**, governed by learnable read/write policies and sparse addressing mechanisms.

The result is a research-oriented framework for experimenting with **memory-centric neural systems**, positioned at the intersection of Transformers, continual learning, and cognitive-inspired AI.

---

## Motivation

Transformers dominate sequence modeling, yet they rely on a fragile assumption:

* All relevant information must fit into a finite context window
* Memory is implicit and distributed
* Forgetting is uncontrolled and often pathological

Real intelligent systems do not work this way.

Humans:

* Store information selectively
* Recall sparsely
* Forget intentionally

This project asks:

* Can we **separate computation from memory**?
* Can memory have its own lifecycle, policies, and constraints?
* Can forgetting be *learned* rather than accidental?

---

## Core Idea

We introduce an architecture composed of four distinct subsystems:

1. **Encoder** - processes the current input.
2. **Semantic Bottleneck** - filters and compresses information before storage.
3. **Multi-Head External Memory** - slot-based persistent storage across time.
4. **Controller** - a Transformer-based brain that decides *what to read*, *what to write*, and *what to forget*.

Memory is not implemented as attention over tokens.

Instead, it is:

* Explicit
* Slot-based
* Sparsely addressed
* Multi-headed

Each forward pass includes **memory I/O operations**, making memory dynamics observable, measurable, and controllable.

---

## Key Features

### 1. Explicit Slot-Based Memory

* Fixed number of memory slots
* Each slot stores a vector embedding
* Memory persists across timesteps and episodes

### 2. Multi-Head Sparse Addressing

* Multiple independent memory heads for parallel access.
* **Top-k cosine similarity** addressing reduces interference and noise.
* Softmax-normalized sparse weights ensure focused updates.

### 3. Semantic Compression (Information Bottleneck)
To prevent memory saturation with noise, we implement an MLP-based bottleneck. It forces the model to extract invariant features before writing, ensuring that each slot stores high-value information.

### 4. Age-based Forgetting (LRU Logic)
Unlike standard MANNs, we track the **age** and **usage** of each slot. The write policy prioritizes overwriting "stale" or "least recently used" slots, preventing memory overflow in extremely long sequences.

### 5. Structural Stability (LayerNorm & Residuals)
To ensure deep gradient flow across thousands of timesteps:
* **LayerNorm** is applied to read vectors to stabilize activation magnitudes.
* **Residual Memory Connections** allow the model to refine its internal state incrementally.

This enables:

* Parallel memory access
* Functional specialization of heads
* Reduced interference

### 6. Learnable Read / Write Policies

Memory updates are controlled by:

* Gated writes
* Residual updates
* Learned erase factors

Writes are *not* forced at every step.

### 7. Differentiable Forgetting

Memory decay is implemented as a learnable process:

* Slot-wise decay gates
* Global memory pressure regularization
* Optional entropy penalties

Forgetting becomes an optimization objective, not a side effect.

---

## Optimization & Training

### Slot Utilization Loss
To prevent **addressing collapse** (where the model uses only a few slots), we optimize for high entropy in memory access:

$$L_{util} = -\sum_{i=1}^{N} \bar{w}_i \log(\bar{w}_i + \epsilon)$$

Where $\bar{w}$ is the average attention weight across heads and batch. This forces the model to explore the entire memory capacity.

### Mixed Precision & Sparsity
The training pipeline supports `torch.cuda.amp` and uses entropy-based penalties to encourage sparse, interpretable memory access patterns.

---

## Architecture Overview

       ┌────────────────────────────────────────────────────────┐
       │                 Input Sequence (Tokens)                │
       └──────────────────────────┬─────────────────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  Transformer Controller   │◄──────────┐
                    │ (Contextual Reasoning)    │           │
                    └──────┬─────────────┬──────┘           │
                           │             │                  │
           ┌───────────────┘             └──────────────┐   │ (Residual 
           ▼                                            ▼   │  Read Vector)
    ┌─────────────┐                             ┌───────────────┐   │
    │ Read Heads  │                             │  Write Heads  │   │
    │ (Top-k Sim) │                             │ (Gated Update)│   │
    └──────┬──────┘                             └───────┬───────┘   │
           │                                            ▼           │
           │                             ┌────────────────────────┐ │
           │                             │   Semantic Bottleneck  │ │
           │                             │ (Information Filter)   │ │
           │                             └──────────────┬─────────┘ │
           ▼                                            ▼           │
    ┌───────────────────────────────────────────────────────────┐   │
    │                 Multi-Head Memory Bank                    │   │
    │      (N Slots × D Dimensions | Age & Usage Buffers)       │   │
    └──────────────────────────┬────────────────────────────────┘   │
                               │                                    │
                               ▼                                    │
                    ┌───────────────────────────┐                   │
                    │   Residual Fusion Layer   ├───────────────────┘
                    │ (Current Read + History)  │
                    └──────────┬────────────────┘
                               ▼
                    ┌───────────────────────────┐
                    │      Prediction Head      │
                    │   (Output Logits / Task)  │
                    └───────────────────────────┘

---

## Implemented Components

* **`MultiHeadMemoryBank`**: Core storage with Top-k addressing and age tracking.
* **`TransformerController`**: A causal Transformer that manages I/O keys and gates.
* **`MemNet`**: The end-to-end model integrating memory and computation.

---

## Experiments Included

### 1. Synthetic Long-Term Dependency Task

**Goal:**
Predict tokens that appeared far outside the local context window.

**Why:**
Tests whether explicit memory outperforms attention-only baselines when context length is constrained.

Metrics:

* Accuracy vs distance
* Memory slot reuse
* Read sparsity

### 2. Continual Learning with Distribution Shifts

**Goal:**
Learn sequential tasks without catastrophic forgetting.

Memory-enabled models:

* Retain task-specific representations
* Exhibit lower forgetting curves

Baseline comparisons:

* Transformer without memory
* RNN/LSTM

### 3. Learned Forgetting Dynamics

**Goal:**
Observe how the model decides *what to forget*.

Tracked signals:

* Slot lifetimes
* Write frequency
* Entropy of read distributions

## Visualization

The repository includes tools for visualizing memory behavior:

* **Heatmaps**: Observe how information travels through slots over time.
* **Slot Survival Curves**: Track which information the model deems "worth remembering" vs "worth forgetting".

---

## Why This Project Is Different

This is **not**:

* Another larger-context Transformer
* A post-hoc interpretability tool
* A retrieval-augmented model glued together

This **is**:

* An architectural rethink
* A research playground for memory systems
* A stepping stone toward continual, adaptive AI

Memory is treated as infrastructure, not an emergent artifact.

---

## Research Directions

This repository is designed to be extended.

Potential research questions:

* Can memory heads specialize functionally?
* What is the optimal memory pressure?
* Can explicit forgetting improve alignment or privacy?
* How does explicit memory interact with planning?

---

## Intended Audience

This project is built for:

* ML / DL Engineers exploring advanced architectures
* Researchers interested in memory, continual learning, or cognitive AI
* Practitioners curious about alternatives to scaling context length

It assumes familiarity with:

* PyTorch
* Attention mechanisms
* Sequence modeling

---

## Status

This is an **experimental research codebase**.

The goal is insight, not benchmark chasing.

Expect:

* Clean abstractions
* Extensive logging
* Easily modifiable components

---

## License

This project is open-source and available under the MIT License.
