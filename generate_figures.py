import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def save(name):
    path = os.path.join("figures", name)

    plt.tight_layout()
    plt.savefig(path, dpi=200)

    print("saved", path)

    plt.close()

def figure_training_loss():
    df = pd.read_csv("results/training_history.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["epoch"],
        df["loss"],
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid(True)

    save("figure_training_loss.png")

def figure_curiosity():
    df = pd.read_csv("results/verification_history.csv")

    train_df = df[df["split"] == "train"]
    ice_df = df[df["split"] == "ice"]

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_df["epoch"],
        train_df["curiosity_score"],
        marker="o",
        label="train"
    )

    plt.plot(
        ice_df["epoch"],
        ice_df["curiosity_score"],
        marker="o",
        label="ice unseen"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Curiosity Score")
    plt.title("Curiosity over Training")

    plt.legend()
    plt.grid(True)

    save("figure_curiosity.png")

def figure_consistency():
    df = pd.read_csv("results/verification_history.csv")

    train_df = df[df["split"] == "train"]
    ice_df = df[df["split"] == "ice"]

    plt.figure(figsize=(10, 5))

    plt.plot(
        train_df["epoch"],
        train_df["consistency_score"],
        marker="o",
        label="train"
    )

    plt.plot(
        ice_df["epoch"],
        ice_df["consistency_score"],
        marker="o",
        label="ice unseen"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Consistency Score")
    plt.title("Consistency over Training")

    plt.legend()
    plt.grid(True)

    save("figure_consistency.png")

def figure_rollout_positions():
    df = pd.read_csv("results/rollout_sample.csv")

    plt.figure(figsize=(10, 5))

    object_ids = sorted(df["object_id"].unique())

    for object_id in object_ids:
        obj = df[df["object_id"] == object_id]

        plt.plot(
            obj["step"],
            obj["model_y"],
            label=f"object {object_id}"
        )

    plt.xlabel("Rollout Step")
    plt.ylabel("Predicted Y Position")
    plt.title("Multi-Object Rollout Positions")

    plt.legend()
    plt.grid(True)

    save("figure_rollout_positions.png")

def figure_rollout_velocity():
    df = pd.read_csv("results/rollout_sample.csv")

    plt.figure(figsize=(10, 5))

    object_ids = sorted(df["object_id"].unique())

    for object_id in object_ids:
        obj = df[df["object_id"] == object_id]

        plt.plot(
            obj["step"],
            obj["model_vy"],
            label=f"object {object_id}"
        )

    plt.xlabel("Rollout Step")
    plt.ylabel("Predicted Vertical Velocity")
    plt.title("Multi-Object Velocity Dynamics")

    plt.legend()
    plt.grid(True)

    save("figure_rollout_velocity.png")

def figure_drift():
    df = pd.read_csv("results/verification_summary.csv")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["curiosity_score"],
        df["final_error"],
        s=80
    )

    plt.xlabel("Curiosity Score")
    plt.ylabel("Final Error")
    plt.title("Curiosity vs Rollout Drift")

    plt.grid(True)

    save("figure_curiosity_vs_drift.png")

figure_training_loss()
figure_curiosity()
figure_consistency()
figure_rollout_positions()
figure_rollout_velocity()
figure_drift()

print()
print("All figures generated in ./figures")