# VVV KaiGraph: A Minimal Relational World Model with Curiosity-Driven Verification

## Abstract

VVV KaiGraph is a compact experimental world-modeling system built around three core ideas: Vectors, Variables, and Visuals. Objects, supports, and relations are represented as typed feature vectors. Scene variables encode dynamic state such as vertical position, velocity, contact, and support area. Visual outputs summarize model behavior through rollouts, verifier scores, and curiosity landscapes.

The system trains a message-passing neural network to predict simple physical transitions from relational scene graphs. It then evaluates multi-step rollouts against an oracle dynamics function and derives a curiosity score from prediction drift, final error, and contact inconsistency. This produces a small but complete loop for relational dynamics prediction, reality verification, and curiosity-driven retraining.

## 1. Motivation

Modern AI systems often excel at static pattern recognition but lack stable internal world models. VVV KaiGraph explores whether a small relational neural system can learn reusable physical concepts from synthetic object-relation scenes and generalize to unseen objects.

The project is intentionally minimal. Instead of building a large simulation stack, it isolates the core ingredients of world modeling:

- symbolic object and relation identity through vectors
- physical state through explicit variables
- interpretable evaluation through visual rollouts and verification metrics

The goal is not to claim broad physical realism. The goal is to create a controllable research prototype for studying relational generalization, rollout drift, and intrinsic curiosity.

## 2. VVV Framework

VVV stands for Vectors, Variables, and Visuals.

### 2.1 Vectors

Objects are represented by six-dimensional feature vectors. The current prototype includes apple, stone, ball, balloon, and ice. These vectors encode properties such as mass, roundness, friction-like factors, rigidity, buoyancy, and object identity indicators.

Supports are also represented as six-dimensional vectors. The current supports include table, floor, sphere, small sphere, and wide plate. Relations are represented as two-dimensional edge vectors for `on` and `in_air`.

### 2.2 Variables

Each object node receives dynamic variables at time `t0`:

- vertical position `y`
- vertical velocity `vy`
- contact state
- estimated support area

The prediction target is a three-dimensional transition vector:

- next vertical position
- next vertical velocity
- next contact probability

### 2.3 Visuals

The system generates visual summaries for architecture, rollout positions, rollout velocities, consistency-curiosity scatter plots, explorer progress, and curiosity heatmaps.

These visuals are not decorative. They are part of the debugging and research process because they reveal attractor states, drift behavior, unseen-object generalization, and high-curiosity regions.

## 3. Model Architecture

KaiDynamicsV04 is a compact graph-style neural dynamics model.

The model contains:

- a node encoder for object and support nodes
- an edge encoder for relation vectors
- a shared message network
- an update network for node state refinement
- a dynamics head predicting `next_y`, `next_vy`, and `next_contact`

The forward pass encodes object node `a`, support node `b`, and relation edge `e`. It computes bidirectional messages, updates both node embeddings, concatenates the updated scene representation, and predicts the next transition.

## 4. Dataset Generation

The dataset is synthetic and generated from object-support-relation triples. Training objects are apple, stone, ball, and balloon. Ice is held out as an unseen test object.

For each sample, the generator randomly selects:

- object type
- support type
- relation type
- initial vertical position
- initial vertical velocity

It then applies a simple oracle rule for stability, gravity, buoyancy, and contact. Stable objects on flat supports maintain contact. Unstable or airborne objects update velocity and position according to gravity and buoyancy.

## 5. Rollout Verification

The trained model is evaluated through multi-step rollouts. At each step, the model predicts the next state, then feeds that prediction back into the next step. This reveals whether the model remains stable over time or accumulates drift.

A separate oracle rollout produces the reference trajectory. The verifier compares model and oracle trajectories across position, velocity, contact, drift growth, and final error.

## 6. Curiosity Score

The current curiosity score is defined as a weighted combination of average position error, average velocity error, average contact error, average drift growth, and final error.

High curiosity indicates scenarios where the model's internal prediction differs strongly from the oracle dynamics. These scenarios are useful because they expose weak regions of the learned world model.

## 7. Explorer Loop

KaiExplorer generates candidate scenarios, scores them through the verifier, selects high-curiosity cases, and retrains the model on those cases.

This forms a minimal self-directed learning loop:

1. generate candidate worlds
2. predict their rollouts
3. compare with oracle reality
4. score prediction drift
5. train on the most surprising scenarios
6. rescore after learning

## 8. Preliminary Results

The current results indicate three notable behaviors.

First, the model learns stable support behavior for ordinary training cases such as apple on table.

Second, the model produces meaningful but imperfect generalization to the held-out object ice, especially when ice appears in relational setups similar to trained objects.

Third, high-curiosity scenarios appear around unstable supports, airborne dynamics, and contact transitions. This suggests that the verifier can identify difficult regions of the relational dynamics space.

The included CSV files summarize rollout positions, rollout velocities, consistency-curiosity scores, and explorer retraining progress.

## 9. Limitations

This prototype is intentionally small and has several limitations.

- The oracle dynamics are hand-designed.
- Object and support vectors are manually specified.
- The current graph contains one object and one support instead of arbitrary multi-object scenes.
- The environment is one-dimensional and does not model horizontal motion, collision geometry, or energy conservation.
- The training loop currently runs as a script rather than a fully configurable experiment runner.

These limitations are useful because they define the next research targets.

## 10. Next Steps

The next development stage should focus on turning the prototype into a cleaner research artifact.

Planned steps:

- separate training, verification, and exploration into independent modules
- add deterministic seeds and experiment configuration
- export metrics automatically into `results/`
- add multi-object graph support
- replace manual vectors with learned embeddings
- introduce recurrent or transformer-based temporal memory
- compare against a non-relational MLP baseline
- add ablation studies for vectors, variables, and message passing

## 11. Repository Structure

Suggested repository layout:

```text
KAI/
  data.py
  model.py
  train.py
  explorer.py
  generate_figures.py
  ResearchNote.md
  requirements.txt
  results/
    curiosity_scan.csv
    rollout_positions.csv
    rollout_velocities.csv
    explorer_progress.csv
  figures/
    figure_1_architecture.png
    figure_2_rollout_positions.png
    figure_3_rollout_velocities.png
    figure_4_consistency_curiosity.png
    figure_5_explorer_progress.png
    figure_6_curiosity_heatmap.png
```

## 12. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python train.py
```

Generate figures:

```bash
python generate_figures.py
```

## 13. Working Title Options

- VVV KaiGraph: Vectors, Variables, and Visuals for Relational World Modeling
- KaiGraph: A Minimal Curiosity-Driven Relational Dynamics Model
- VVV: A Compact Framework for Verifiable Neural World Models

## 14. Current Status

This is an early research prototype. It is suitable for a GitHub research note, preprint preparation, and further experiments. It should not yet be presented as a complete physical simulator or general AGI system.
