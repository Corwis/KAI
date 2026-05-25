import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import KaiDynamicsDataset
from model import KaiDynamicsV04
from train import (
    predict_case,
    rollout,
    verify_rollout,
    curiosity_scan,
    KaiExplorer
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)

def train_model(train_size, epochs):
    dataset = KaiDynamicsDataset(
        size=train_size,
        use_test_object=False
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = KaiDynamicsV04()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(epochs):
        total_loss = 0

        for node_a, edge, node_b, target, obj, relation, support in loader:
            optimizer.zero_grad()

            prediction, _, _ = model(node_a, edge, node_b)

            loss = loss_fn(prediction, target)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        history.append([epoch, total_loss])

        if epoch % 50 == 0:
            print("epoch", epoch, "loss", total_loss)

    return model, optimizer, loss_fn, history

def export_training_history(output_dir, history):
    write_csv(
        os.path.join(output_dir, "training_history.csv"),
        ["epoch", "loss"],
        history
    )

def export_verifications(output_dir):
    cases = [
        ("apple", "on", "table"),
        ("apple", "on", "sphere"),
        ("ice", "on", "sphere"),
        ("balloon", "in_air", "floor")
    ]

    rows = []

    for obj, relation, support in cases:
        result = verify_rollout(
            obj,
            relation,
            support,
            steps=50
        )

        rows.append([
            obj,
            relation,
            support,
            result["avg_position_error"],
            result["avg_velocity_error"],
            result["avg_contact_error"],
            result["avg_drift_growth"],
            result["final_error"],
            result["curiosity_score"],
            result["consistency_score"]
        ])

    write_csv(
        os.path.join(output_dir, "verification_results.csv"),
        [
            "object",
            "relation",
            "support",
            "avg_position_error",
            "avg_velocity_error",
            "avg_contact_error",
            "avg_drift_growth",
            "final_error",
            "curiosity_score",
            "consistency_score"
        ],
        rows
    )

def export_rollouts(output_dir):
    scenarios = [
        ("apple", "on", "sphere"),
        ("ice", "on", "sphere"),
        ("balloon", "in_air", "floor")
    ]

    rows = []

    for obj, relation, support in scenarios:
        trajectory = rollout(
            obj,
            relation,
            support,
            steps=15,
            verbose=False
        )

        for step, item in enumerate(trajectory):
            rows.append([
                obj,
                relation,
                support,
                step,
                item["y"],
                item["vy"],
                item["contact"]
            ])

    write_csv(
        os.path.join(output_dir, "rollouts.csv"),
        [
            "object",
            "relation",
            "support",
            "step",
            "y",
            "vy",
            "contact"
        ],
        rows
    )

def save_model(output_dir, model):
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "kai_model.pt")
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    set_seed(args.seed)

    ensure_dir(args.output_dir)

    model, optimizer, loss_fn, history = train_model(
        train_size=args.train_size,
        epochs=args.epochs
    )

    export_training_history(
        args.output_dir,
        history
    )

    export_verifications(args.output_dir)

    export_rollouts(args.output_dir)

    save_model(args.output_dir, model)

    print()
    print("Results exported to", args.output_dir)

if __name__ == "__main__":
    main()