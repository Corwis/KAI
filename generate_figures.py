import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def save(name):
    path = os.path.join("figures", name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved", path)

def figure_1_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    boxes = [
        ("Scene Graph\nObjects + Relations", 0.5, 0.88),
        ("Relation Encoder\nNode + Edge Features", 0.5, 0.72),
        ("Message Passing\nRelational Latent Space", 0.5, 0.56),
        ("Dynamics Predictor\nnext_y, next_vy, contact", 0.5, 0.40),
        ("Rollout\nscene_t0 → scene_t1 → scene_t2", 0.5, 0.24),
        ("Prediction Drift\nReality Verification + Curiosity", 0.5, 0.08)
    ]

    for text, x, y in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black")
        )

    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(0.5, boxes[i + 1][2] + 0.06),
            xytext=(0.5, boxes[i][2] - 0.06),
            arrowprops=dict(arrowstyle="->", linewidth=1.5)
        )

    ax.set_title("KaiGraph / VVV Relational World Model", fontsize=16)
    save("figure_1_architecture.png")

def figure_2_rollouts():
    steps = np.arange(15)

    apple_y = [0.8926, 0.7167, 0.549, 0.4581, 0.4307, 0.4227, 0.4203, 0.4196, 0.4194, 0.4194, 0.4194, 0.4193, 0.4193, 0.4193, 0.4193]
    ice_y = [0.8208, 0.6259, 0.4762, 0.4269, 0.4108, 0.4056, 0.4039, 0.4034, 0.4033, 0.4032, 0.4032, 0.4032, 0.4032, 0.4032, 0.4032]
    balloon_y = [1.0991, 1.2761, 1.5368, 1.7905, 2.0156, 2.186, 2.2942, 2.356, 2.3821, 2.3845, 2.3742, 2.3589, 2.3434, 2.3304, 2.3209]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, apple_y, marker="o", label="apple on sphere")
    plt.plot(steps, ice_y, marker="o", label="ice on sphere unseen")
    plt.plot(steps, balloon_y, marker="o", label="balloon in_air")

    plt.xlabel("Rollout step")
    plt.ylabel("Predicted y position")
    plt.title("Multi-Step Rollouts: Attractors and Emergent Dynamics")
    plt.legend()
    plt.grid(True)

    save("figure_2_rollout_positions.png")

def figure_3_velocities():
    steps = np.arange(15)

    apple_vy = [-0.1045, -0.1723, -0.1707, -0.1513, -0.1459, -0.1447, -0.1444, -0.1444, -0.1444, -0.1444, -0.1444, -0.1444, -0.1444, -0.1444, -0.1444]
    ice_vy = [-0.1054, -0.1632, -0.1536, -0.1496, -0.1488, -0.1487, -0.1488, -0.1488, -0.1488, -0.1488, -0.1488, -0.1488, -0.1488, -0.1488, -0.1488]
    balloon_vy = [0.0785, 0.1442, 0.1743, 0.1891, 0.1895, 0.1799, 0.1666, 0.1521, 0.1391, 0.1286, 0.1213, 0.1168, 0.1145, 0.1139, 0.114]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, apple_vy, marker="o", label="apple on sphere")
    plt.plot(steps, ice_vy, marker="o", label="ice on sphere unseen")
    plt.plot(steps, balloon_vy, marker="o", label="balloon in_air")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Rollout step")
    plt.ylabel("Predicted vertical velocity")
    plt.title("Velocity Dynamics: Falling vs Rising Behaviors")
    plt.legend()
    plt.grid(True)

    save("figure_3_rollout_velocities.png")

def figure_4_curiosity_scatter():
    cases = [
        ("apple table", 0.90776, 0.10162),
        ("ice table", 0.33654, 1.97145),
        ("apple sphere", 0.00577, 172.46006),
        ("ice sphere", 0.00577, 172.31),
        ("stone air", 0.00575, 172.92455),
        ("ice air", 0.00578, 171.9034),
        ("balloon sphere", 0.00859, 115.41042),
        ("balloon air", 0.00876, 113.17129)
    ]

    x = [c[1] for c in cases]
    y = [c[2] for c in cases]

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=80)

    for label, consistency, curiosity in cases:
        plt.annotate(label, (consistency, curiosity), textcoords="offset points", xytext=(6, 6))

    plt.xlabel("Consistency score")
    plt.ylabel("Curiosity score")
    plt.title("Reality Verifier: Consistency vs Curiosity")
    plt.grid(True)

    save("figure_4_consistency_curiosity.png")

def figure_5_explorer_progress():
    iterations = np.array([0, 1, 2])

    before = np.array([174.7975, 174.7773, 174.5305])
    after = np.array([171.693, 172.2727, 172.2094])

    plt.figure(figsize=(9, 5))
    plt.plot(iterations, before, marker="o", label="before retraining")
    plt.plot(iterations, after, marker="o", label="after retraining")

    plt.xlabel("Explorer iteration")
    plt.ylabel("Top scenario curiosity")
    plt.title("KaiExplorer v0.7: Curiosity-Driven Retraining")
    plt.legend()
    plt.grid(True)

    save("figure_5_explorer_progress.png")

def figure_6_curiosity_heatmap():
    object_roundness = np.linspace(0, 1, 25)
    support_roundness = np.linspace(0, 1, 25)

    heatmap = np.zeros((len(object_roundness), len(support_roundness)))

    for i, obj_r in enumerate(object_roundness):
        for j, sup_r in enumerate(support_roundness):
            support_area = max(0.05, (1.0 - sup_r) * 0.3)
            instability = sup_r * 0.9 + obj_r * 0.15 - support_area * 0.8
            curiosity = max(0, instability) * 180
            heatmap[i, j] = curiosity

    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto"
    )

    plt.colorbar(label="Estimated curiosity")
    plt.xlabel("Support roundness")
    plt.ylabel("Object roundness")
    plt.title("Curiosity Landscape over Relational Geometry")

    save("figure_6_curiosity_heatmap.png")

figure_1_architecture()
figure_2_rollouts()
figure_3_velocities()
figure_4_curiosity_scatter()
figure_5_explorer_progress()
figure_6_curiosity_heatmap()

print()
print("All figures generated in ./figures")