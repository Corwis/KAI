import argparse
import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    KaiMultiObjectDataset,
    generate_scene,
    apply_physics_step,
    build_edges,
    NODE_DIM,
    EDGE_DIM,
    TARGET_DIM
)
from model import KaiDynamics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def masked_mse(prediction, target, node_mask):
    mask = node_mask.unsqueeze(-1)
    loss = (prediction - target) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)

def target_to_nodes(nodes, prediction, node_mask):
    next_nodes = []

    for i, node in enumerate(nodes):
        updated = node[:]

        if node_mask[i] == 0.0:
            next_nodes.append(updated)
            continue

        updated[7] = float(prediction[i][0])
        updated[8] = float(prediction[i][1])
        updated[9] = float(prediction[i][2])
        updated[10] = float(prediction[i][3])
        updated[11] = float(prediction[i][4])
        updated[12] = float((updated[9] ** 2 + updated[10] ** 2) ** 0.5)
        updated[13] = 1.0

        next_nodes.append(updated)

    return next_nodes

def model_step(model, nodes, node_mask, max_objects, device):
    edges = build_edges(nodes, node_mask, max_objects)

    node_tensor = torch.tensor([nodes], dtype=torch.float32).to(device)
    edge_tensor = torch.tensor([edges], dtype=torch.float32).to(device)
    mask_tensor = torch.tensor([node_mask], dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction, hidden = model(node_tensor, edge_tensor, mask_tensor)

    return prediction.cpu().numpy()[0].tolist()

def model_rollout(model, start_nodes, node_mask, steps, max_objects, device):
    model.eval()

    nodes = [node[:] for node in start_nodes]
    trajectory = []

    for _ in range(steps):
        pred = model_step(
            model=model,
            nodes=nodes,
            node_mask=node_mask,
            max_objects=max_objects,
            device=device
        )

        trajectory.append(pred)
        nodes = target_to_nodes(nodes, pred, node_mask)

    return trajectory

def oracle_rollout(start_nodes, node_mask, steps, max_objects):
    nodes = [node[:] for node in start_nodes]
    trajectory = []

    for _ in range(steps):
        next_nodes, targets = apply_physics_step(nodes, node_mask, max_objects)
        trajectory.append(targets)
        nodes = next_nodes

    return trajectory

def verify_scene(model, steps, max_objects, device, use_test_object=False):
    nodes, edges, targets, node_mask, next_nodes = generate_scene(
        min_objects=3,
        max_objects=max_objects,
        use_test_object=use_test_object
    )

    model_traj = model_rollout(
        model=model,
        start_nodes=nodes,
        node_mask=node_mask,
        steps=steps,
        max_objects=max_objects,
        device=device
    )

    oracle_traj = oracle_rollout(
        start_nodes=nodes,
        node_mask=node_mask,
        steps=steps,
        max_objects=max_objects
    )

    position_errors = []
    velocity_errors = []
    contact_errors = []
    drift_growth = []

    last_step_error = 0.0

    for step in range(steps):
        step_error = 0.0

        for i in range(max_objects):
            if node_mask[i] == 0.0:
                continue

            px = model_traj[step][i][0]
            py = model_traj[step][i][1]
            pvx = model_traj[step][i][2]
            pvy = model_traj[step][i][3]
            pc = model_traj[step][i][4]

            ox = oracle_traj[step][i][0]
            oy = oracle_traj[step][i][1]
            ovx = oracle_traj[step][i][2]
            ovy = oracle_traj[step][i][3]
            oc = oracle_traj[step][i][4]

            pos_error = abs(px - ox) + abs(py - oy)
            vel_error = abs(pvx - ovx) + abs(pvy - ovy)
            contact_error = abs(pc - oc)

            position_errors.append(pos_error)
            velocity_errors.append(vel_error)
            contact_errors.append(contact_error)

            step_error += pos_error + vel_error + contact_error

        drift_growth.append(max(0.0, step_error - last_step_error))
        last_step_error = step_error

    avg_position_error = sum(position_errors) / max(1, len(position_errors))
    avg_velocity_error = sum(velocity_errors) / max(1, len(velocity_errors))
    avg_contact_error = sum(contact_errors) / max(1, len(contact_errors))
    avg_drift_growth = sum(drift_growth) / max(1, len(drift_growth))
    final_error = last_step_error

    curiosity_score = (
        avg_position_error * 1.5
        + avg_velocity_error * 2.0
        + avg_contact_error * 1.0
        + avg_drift_growth * 3.0
        + final_error * 0.25
    )

    consistency_score = 1.0 / (1.0 + curiosity_score)

    return {
        "objects": int(sum(node_mask)),
        "avg_position_error": avg_position_error,
        "avg_velocity_error": avg_velocity_error,
        "avg_contact_error": avg_contact_error,
        "avg_drift_growth": avg_drift_growth,
        "final_error": final_error,
        "curiosity_score": curiosity_score,
        "consistency_score": consistency_score
    }

def export_rollout_sample(model, output_dir, max_objects, device, steps=100):
    nodes, edges, targets, node_mask, next_nodes = generate_scene(
        min_objects=3,
        max_objects=max_objects,
        use_test_object=False
    )

    model_traj = model_rollout(
        model=model,
        start_nodes=nodes,
        node_mask=node_mask,
        steps=steps,
        max_objects=max_objects,
        device=device
    )

    oracle_traj = oracle_rollout(
        start_nodes=nodes,
        node_mask=node_mask,
        steps=steps,
        max_objects=max_objects
    )

    rows = []

    for step in range(steps):
        for i in range(max_objects):
            if node_mask[i] == 0.0:
                continue

            rows.append([
                step,
                i,
                model_traj[step][i][0],
                model_traj[step][i][1],
                model_traj[step][i][2],
                model_traj[step][i][3],
                model_traj[step][i][4],
                oracle_traj[step][i][0],
                oracle_traj[step][i][1],
                oracle_traj[step][i][2],
                oracle_traj[step][i][3],
                oracle_traj[step][i][4]
            ])

    write_csv(
        os.path.join(output_dir, "rollout_sample.csv"),
        [
            "step",
            "object_id",
            "model_x",
            "model_y",
            "model_vx",
            "model_vy",
            "model_contact",
            "oracle_x",
            "oracle_y",
            "oracle_vx",
            "oracle_vy",
            "oracle_contact"
        ],
        rows
    )

def train(args):
    set_seed(args.seed)

    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = KaiMultiObjectDataset(
        size=args.train_size,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        use_test_object=False
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = KaiDynamics(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden=args.hidden,
        message_steps=args.message_steps,
        target_dim=TARGET_DIM
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    history_rows = []
    verification_rows = []

    print()
    print("KAI MULTI-OBJECT DYNAMICS")
    print("-------------------------")
    print("device", device)
    print("train_size", args.train_size)
    print("objects_per_scene", str(args.min_objects) + "-" + str(args.max_objects))
    print("node_dim", NODE_DIM)
    print("edge_dim", EDGE_DIM)
    print("message_steps", args.message_steps)
    print()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for nodes, edges, targets, node_mask, next_nodes in loader:
            nodes = nodes.to(device)
            edges = edges.to(device)
            targets = targets.to(device)
            node_mask = node_mask.to(device)

            optimizer.zero_grad()

            prediction, hidden = model(nodes, edges, node_mask)

            loss = masked_mse(prediction, targets, node_mask)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += float(loss.item())

        history_rows.append([epoch, total_loss])

        if epoch % args.log_every == 0:
            train_result = verify_scene(
                model=model,
                steps=args.verify_steps,
                max_objects=args.max_objects,
                device=device,
                use_test_object=False
            )

            ice_result = verify_scene(
                model=model,
                steps=args.verify_steps,
                max_objects=args.max_objects,
                device=device,
                use_test_object=True
            )

            verification_rows.append([
                epoch,
                "train",
                train_result["objects"],
                train_result["avg_position_error"],
                train_result["avg_velocity_error"],
                train_result["avg_contact_error"],
                train_result["avg_drift_growth"],
                train_result["final_error"],
                train_result["curiosity_score"],
                train_result["consistency_score"]
            ])

            verification_rows.append([
                epoch,
                "ice",
                ice_result["objects"],
                ice_result["avg_position_error"],
                ice_result["avg_velocity_error"],
                ice_result["avg_contact_error"],
                ice_result["avg_drift_growth"],
                ice_result["final_error"],
                ice_result["curiosity_score"],
                ice_result["consistency_score"]
            ])

            print("epoch", epoch, "loss", round(total_loss, 6))

            print(
                "train",
                "objects",
                train_result["objects"],
                "curiosity",
                round(train_result["curiosity_score"], 5),
                "consistency",
                round(train_result["consistency_score"], 5),
                "final_error",
                round(train_result["final_error"], 5)
            )

            print(
                "ice",
                "objects",
                ice_result["objects"],
                "curiosity",
                round(ice_result["curiosity_score"], 5),
                "consistency",
                round(ice_result["consistency_score"], 5),
                "final_error",
                round(ice_result["final_error"], 5)
            )

            print()

    model_path = os.path.join(args.output_dir, "kai_multi_object.pt")

    torch.save(
        model.state_dict(),
        model_path
    )

    write_csv(
        os.path.join(args.output_dir, "training_history.csv"),
        ["epoch", "loss"],
        history_rows
    )

    write_csv(
        os.path.join(args.output_dir, "verification_history.csv"),
        [
            "epoch",
            "split",
            "objects",
            "avg_position_error",
            "avg_velocity_error",
            "avg_contact_error",
            "avg_drift_growth",
            "final_error",
            "curiosity_score",
            "consistency_score"
        ],
        verification_rows
    )

    export_rollout_sample(
        model=model,
        output_dir=args.output_dir,
        max_objects=args.max_objects,
        device=device,
        steps=args.long_steps
    )

    long_result = verify_scene(
        model=model,
        steps=args.long_steps,
        max_objects=args.max_objects,
        device=device,
        use_test_object=False
    )

    print("FINAL 100 STEP STABILITY")
    print("------------------------")
    print("objects", long_result["objects"])
    print("avg_position_error", round(long_result["avg_position_error"], 6))
    print("avg_velocity_error", round(long_result["avg_velocity_error"], 6))
    print("avg_contact_error", round(long_result["avg_contact_error"], 6))
    print("avg_drift_growth", round(long_result["avg_drift_growth"], 6))
    print("final_error", round(long_result["final_error"], 6))
    print("curiosity_score", round(long_result["curiosity_score"], 6))
    print("consistency_score", round(long_result["consistency_score"], 6))
    print()
    print("saved", model_path)
    print("exported", args.output_dir)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message-steps", type=int, default=4)
    parser.add_argument("--min-objects", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--verify-steps", type=int, default=30)
    parser.add_argument("--long-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()