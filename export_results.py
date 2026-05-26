import os
import csv
import argparse
import random
import numpy as np
import torch

from data import (
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

def compute_metrics(model_traj, oracle_traj, node_mask, steps, max_objects):
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
        "avg_position_error": avg_position_error,
        "avg_velocity_error": avg_velocity_error,
        "avg_contact_error": avg_contact_error,
        "avg_drift_growth": avg_drift_growth,
        "final_error": final_error,
        "curiosity_score": curiosity_score,
        "consistency_score": consistency_score
    }

def export_rollouts(model, output_dir, device, scenes, steps, max_objects):
    rollout_rows = []
    summary_rows = []

    for scene_id in range(scenes):
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

        metrics = compute_metrics(
            model_traj=model_traj,
            oracle_traj=oracle_traj,
            node_mask=node_mask,
            steps=steps,
            max_objects=max_objects
        )

        summary_rows.append([
            scene_id,
            int(sum(node_mask)),
            metrics["avg_position_error"],
            metrics["avg_velocity_error"],
            metrics["avg_contact_error"],
            metrics["avg_drift_growth"],
            metrics["final_error"],
            metrics["curiosity_score"],
            metrics["consistency_score"]
        ])

        for step in range(steps):
            for object_id in range(max_objects):
                if node_mask[object_id] == 0.0:
                    continue

                rollout_rows.append([
                    scene_id,
                    step,
                    object_id,
                    model_traj[step][object_id][0],
                    model_traj[step][object_id][1],
                    model_traj[step][object_id][2],
                    model_traj[step][object_id][3],
                    model_traj[step][object_id][4],
                    oracle_traj[step][object_id][0],
                    oracle_traj[step][object_id][1],
                    oracle_traj[step][object_id][2],
                    oracle_traj[step][object_id][3],
                    oracle_traj[step][object_id][4]
                ])

    write_csv(
        os.path.join(output_dir, "rollouts.csv"),
        [
            "scene_id",
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
        rollout_rows
    )

    write_csv(
        os.path.join(output_dir, "verification_summary.csv"),
        [
            "scene_id",
            "objects",
            "avg_position_error",
            "avg_velocity_error",
            "avg_contact_error",
            "avg_drift_growth",
            "final_error",
            "curiosity_score",
            "consistency_score"
        ],
        summary_rows
    )

def load_model(model_path, device, hidden, message_steps):
    model = KaiDynamics(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden=hidden,
        message_steps=message_steps,
        target_dim=TARGET_DIM
    ).to(device)

    state = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state)
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="results/kai_multi_object.pt")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--message-steps", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_model(
        model_path=args.model,
        device=device,
        hidden=args.hidden,
        message_steps=args.message_steps
    )

    export_rollouts(
        model=model,
        output_dir=args.output_dir,
        device=device,
        scenes=args.scenes,
        steps=args.steps,
        max_objects=args.max_objects
    )

    print()
    print("exported", args.output_dir)
    print("rollouts.csv")
    print("verification_summary.csv")

if __name__ == "__main__":
    main()