# KaiGraph / VVV: Minimal Relational World Models

**Emergent Dynamics, Rollout Consistency, and Intrinsic Curiosity via Prediction Drift**

A minimal relational world model exploring whether physical-like dynamics can emerge from relational interactions rather than explicit rules.

## Initial Results

- ✅ **Emergent physics** without hardcoded equations (falling, buoyancy, stability)
- ✅ **Generalization to unseen objects** via relational embeddings
- ✅ **Multi-step rollout stability** with dynamic attractors
- ✅ **Intrinsic curiosity** emerging from prediction drift
- ✅ **Self-directed learning** via curiosity-driven retraining

---

## Quick Start

```bash
git clone https://github.com/corwis/kaigraph.git
cd kaigraph

pip install -r requirements.txt

python train.py              # Train the model
python generate_figures.py   # Generate publication figures
python explorer.py           # Run active learning loop
```

---

## Core Concept

Instead of predicting tokens, KaiGraph predicts evolving **relational world states**.

```
Scene Graph (Objects + Relations)
    ↓
Relational Encoding (Node + Edge Features)
    ↓
Message Passing (Graph Neural Network)
    ↓
Latent Dynamics Learning
    ↓
Multi-Step Rollouts
    ↓
Prediction Drift Analysis
    ↓
Curiosity Scoring
    ↓
Self-Directed Retraining
```

---

## Key Experimental Results

### 1. Emergent Relational Dynamics

Without explicit physics rules, the system learns:
- **Falling behavior** on unstable supports (negative velocity)
- **Rising behavior** for buoyant objects (positive velocity)  
- **Stable equilibrium** on flat surfaces (near-zero velocity)

### 2. Generalization to Unseen Objects

Object `ice` was **never shown during training**.

```
apple on sphere → vy ≈ -0.1655
ice on sphere   → vy ≈ -0.1737
Difference: 4.7%
```

The model generalizes through **relational structure**, not object identity.

### 3. Multi-Step Rollout Stability

Rollouts converge to dynamic attractors, not chaos:

```
apple on sphere:
step 0:  y=0.89
step 1:  y=0.71
step 2:  y=0.54
step 3:  y=0.42
step 4+: y≈0.42 (STABLE ATTRACTOR)
```

### 4. Intrinsic Curiosity via Prediction Drift

Prediction drift automatically identifies what the model doesn't understand:

| Scenario | Consistency | Curiosity | Interpretation |
|----------|-------------|-----------|-----------------|
| apple on table | 0.91 | 0.10 | Well-understood (boring) |
| apple on sphere | 0.006 | 172+ | Poorly-understood (interesting) |
| ice on sphere | 0.006 | 173 | High-priority learning target |

No external reward signal needed.

### 5. Self-Directed Learning (KaiExplorer v0.7)

The model generates scenario variations, evaluates curiosity, and retrains on high-curiosity states:

```
Before retraining: ball on sphere → curiosity 174.79
After retraining:  ball on sphere → curiosity 171.69 ↓
```

Measurable self-improvement without external guidance.

---

## Architecture

### Input Features

**Node (Object) Features:**
```
mass, roundness, flatness, rigidity, buoyancy, movable, y, vy, contact
```

**Edge (Relation) Features:**
```
relation_type, contact_area, support_strength
```

### Model Components

- **Relation Encoder** - Processes node and edge features
- **Message Passing** - Graph neural network creates 8D latent embeddings
- **Dynamics Predictor** - Outputs next_y, next_vy, next_contact
- **Rollout Verifier** - Computes consistency and curiosity metrics
- **Explorer Loop** - Generates and selects training scenarios

---

## Experimental Setup

### Experiment 1: Generalization
- **Train:** apple, stone on table/sphere
- **Test:** unseen object (ice) on same surfaces
- **Result:** Correct generalization to unseen objects

### Experiment 2: Rollout Stability  
- **Test:** 15-step rollouts from various starting states
- **Result:** Convergence to stable attractors (no explosion)

### Experiment 3: Emergent Physics
- **Question:** Can falling/rising emerge without explicit physics?
- **Result:** Yes. Behaviors emerge from relational message passing.

### Experiment 4: Curiosity-Driven Retraining
- **Question:** Does prediction drift guide learning priorities?
- **Result:** High-drift scenarios improve faster than low-drift

---

## Publication Figures

Generated with `python generate_figures.py`:

1. **Architecture Diagram** - System overview
2. **Rollout Trajectories** - Position over time  
3. **Velocity Dynamics** - Falling vs rising behaviors
4. **Consistency vs Curiosity** - Scatter plot showing learned understanding
5. **Explorer Progress** - Self-improvement iterations
6. **Curiosity Heatmap** - Landscape of learning difficulty

All figures included in `/figures/` directory.

---

## Limitations (Intentional)

This is a **minimal research prototype**, not a production system.

- **Synthetic data only** - Parametrically generated, not real physics
- **1D vertical dynamics** - Simplified to test core concepts
- **No collisions** - Multi-object interactions not implemented
- **Hand-engineered features** - Features manually defined (not learned)
- **No vision** - Works on symbolic scene graphs only
- **No intervention** - Observations only; model cannot act on world
- **Small scale** - ~100 training scenarios
- **Short horizon** - 15-step rollouts tested

These limitations define the research frontier, not failures.

---

## Repository Structure

```
kaigraph/
├── train.py                 # Main training script
├── model.py                 # KaiGraph architecture
├── verifier.py              # Curiosity + consistency metrics
├── explorer.py              # KaiExplorer v0.7 (active learning)
├── data.py                  # Dataset generation
├── generate_figures.py      # Publication figure generation
│
├── figures/                 # Output figures
│   ├── figure_1_architecture.png
│   ├── figure_2_rollout_positions.png
│   ├── figure_3_rollout_velocities.png
│   ├── figure_4_consistency_curiosity.png
│   ├── figure_5_explorer_progress.png
│   └── figure_6_curiosity_heatmap.png
│
├── results/                 # Training outputs
│   ├── rollouts.csv
│   ├── curiosity_scores.csv
│   └── explorer_log.txt
│
├── paper/
│   └── KaiGraph_ResearchNote.md
│
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

---

## Next Steps

### Short Term (v0.2)
- Multi-object dynamics
- Collision detection
- Longer rollouts (100+ steps)
- Learned feature extraction

### Medium Term (v0.5)
- Vision → scene graph conversion
- Active intervention learning  
- Larger synthetic environments

### Long Term (v1.0)
- Real-world simulator comparisons
- Robotics experiments
- Hierarchical world models

---

## Citation

```bibtex
@misc{kaigraph2026,
  title={Minimal Relational World Models: Emergent Dynamics, 
         Rollout Consistency, and Intrinsic Curiosity 
         via Prediction Drift},
  author={Corwis},
  year={2026},
  howpublished={\url{https://github.com/corwis/kaigraph}},
  note={Research Preprint}
}
```

---

## References

- Ha & Schmidhuber (2018) - World Models
- Battaglia et al. (2018) - Relational Inductive Biases, Deep Learning, and Graph Networks
- Pathak et al. (2017) - Curiosity-driven Exploration by Self-supervised Prediction
- Kipf et al. (2019) - Contrastive Learning of Structured World Models

---

## License

MIT License - Non-commercial research use

---

## Status

**Experimental Research Prototype — v0.1**

**Author:** Corwis  
**Last Updated:** 2026-05-25  

For questions or feedback, open an issue on GitHub.
