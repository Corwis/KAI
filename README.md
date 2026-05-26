# KaiGraph

**A Minimal Relational World Model for Multi-Object Dynamics, Rollout Verification, and Curiosity-Driven Learning**

KaiGraph is an experimental graph-based world model that explores whether physical-like behavior can emerge from relational interactions between objects.

Instead of predicting tokens, KaiGraph predicts evolving relational world states.

The system combines:

* graph neural message passing
* multi-object dynamics prediction
* rollout verification
* intrinsic curiosity
* self-directed exploration

---

# Core Idea

A scene is represented as a relational graph:

```text
Objects (nodes)
+
Object-to-object relations (edges)
```

The model predicts how the entire scene evolves over time.

```text
Scene Graph
    ↓
Node + Edge Encoding
    ↓
Message Passing
    ↓
Latent Relational Dynamics
    ↓
Multi-Step Rollouts
    ↓
Prediction Drift
    ↓
Curiosity Scoring
```

---

# Current Features

## Multi-Object Dynamics

KaiGraph supports dynamic scenes with multiple interacting objects.

Examples:

* falling objects
* buoyancy
* stacking-like behavior
* contact transitions
* collision-like interactions
* rollout stability

Scenes contain:

* 3-5 objects
* pairwise relations
* dynamic interaction graphs

---

## Graph Neural Message Passing

Every object exchanges information with every other object.

```text
object ↔ object
```

The model learns relational interactions through iterative message passing instead of hardcoded symbolic rules.

---

## Rollout Verification

The model performs long-horizon rollouts and compares them against an oracle dynamics system.

Metrics include:

* position error
* velocity error
* contact error
* drift growth
* final rollout error
* consistency score
* curiosity score

---

## Intrinsic Curiosity

Curiosity emerges from prediction drift.

Scenarios with unstable or poorly-understood dynamics automatically receive higher curiosity scores.

This allows the system to identify:

* unstable interactions
* rollout failure regions
* weak internal world models

without external rewards.

---

## Self-Directed Exploration

`KaiExplorer` generates candidate scenes, evaluates curiosity, and retrains on high-curiosity scenarios.

This creates a minimal autonomous learning loop:

```text
generate scenes
    ↓
predict rollouts
    ↓
compare with oracle
    ↓
measure drift
    ↓
select surprising worlds
    ↓
retrain
```

---

# Repository Structure

```text
KAI/
│
├── data.py
├── model.py
├── train.py
├── explorer.py
├── export_results.py
├── generate_figures.py
│
├── README.md
├── ResearchNote.md
├── requirements.txt
│
├── results/
│   ├── training_history.csv
│   ├── verification_history.csv
│   ├── rollout_sample.csv
│   ├── rollouts.csv
│   └── verification_summary.csv
│
└── figures/
    ├── figure_training_loss.png
    ├── figure_curiosity.png
    ├── figure_consistency.png
    ├── figure_rollout_positions.png
    ├── figure_rollout_velocity.png
    └── figure_curiosity_vs_drift.png
```

---

# Installation

```bash
git clone https://github.com/Corwis/KAI.git
cd KAI

pip install -r requirements.txt
```

---

# Training

Train the relational dynamics model:

```bash
python train.py
```

Outputs:

* trained model checkpoint
* rollout metrics
* verification metrics
* rollout samples

saved into:

```text
results/
```

---

# Export Results

Generate rollout exports and verification summaries:

```bash
python export_results.py
```

---

# Generate Figures

Generate research figures from CSV results:

```bash
python generate_figures.py
```

Generated figures are saved into:

```text
figures/
```

---

# Example Concepts Learned

The current prototype can learn simplified forms of:

* gravity-like behavior
* buoyancy
* contact transitions
* object interactions
* rollout stabilization
* relational generalization

without explicit symbolic physics equations.

---

# Research Direction

KaiGraph explores the idea that future AI systems may require:

```text
Language Models
+
World Models
+
Verification
+
Planning
+
Persistent State
```

instead of pure token prediction systems.

The project focuses on:

* relational reasoning
* predictive simulation
* rollout consistency
* emergent dynamics
* curiosity-driven learning

---

# Limitations

This is an early research prototype.

Current limitations:

* simplified synthetic physics
* no real collision solver
* no rotational dynamics
* no transformer memory
* small-scale environments
* synthetic oracle dynamics
* limited rollout stability

The goal is not realistic physics simulation.

The goal is studying:

* relational dynamics learning
* world modeling
* prediction drift
* intrinsic curiosity

in minimal systems.

---

# Future Work

Planned directions:

* larger multi-object scenes
* learned embeddings
* temporal memory
* attention-based interactions
* transformer world models
* adaptive rollout horizons
* differentiable planning
* language-conditioned world states
* multimodal integration
* agentic reasoning systems

---

# Status

Current status:

* working research prototype
* active experimentation
* multi-object relational dynamics operational
* rollout verification operational
* curiosity-driven exploration operational

---

# License

MIT License
