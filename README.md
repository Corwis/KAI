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

### Model Component
