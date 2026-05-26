import random
import torch
from torch.utils.data import Dataset, DataLoader

from data import (
    generate_scene,
    apply_physics_step,
    build_edges,
    NODE_DIM,
    EDGE_DIM,
    TARGET_DIM
)

class ExplorerDataset(Dataset):
    def __init__(self, scenes):
        self.samples = []

        for nodes, edges, targets, node_mask, next_nodes in scenes:
            self.samples.append((
                torch.tensor(nodes, dtype=torch.float32),
                torch.tensor(edges, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(node_mask, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class KaiExplorer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.memory = []

    def generate_candidates(self, count=40, max_objects=5):
        candidates = []

        for _ in range(count):
            scene = generate_scene(
                min_objects=3,
                max_objects=max_objects,
                use_test_object=random.random() < 0.3
            )

            candidates.append(scene)

        return candidates

    def rollout_step(self, nodes, node_mask, max_objects):
        edges = build_edges(nodes, node_mask, max_objects)

        node_tensor = torch.tensor([nodes], dtype=torch.float32).to(self.device)
        edge_tensor = torch.tensor([edges], dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor([node_mask], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction, hidden = self.model(
                node_tensor,
                edge_tensor,
                mask_tensor
            )

        return prediction.cpu().numpy()[0].tolist()

    def prediction_to_nodes(self, nodes, prediction, node_mask):
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

            next_nodes.append(updated)

        return next_nodes

    def curiosity_score(self, nodes, node_mask, max_objects, rollout_steps=30):
        model_nodes = [n[:] for n in nodes]
        oracle_nodes = [n[:] for n in nodes]

        total_error = 0.0
        drift_growth = 0.0
        previous_error = 0.0

        for _ in range(rollout_steps):
            prediction = self.rollout_step(
                model_nodes,
                node_mask,
                max_objects
            )

            model_nodes = self.prediction_to_nodes(
                model_nodes,
                prediction,
                node_mask
            )

            oracle_nodes, oracle_targets = apply_physics_step(
                oracle_nodes,
                node_mask,
                max_objects
            )

            step_error = 0.0

            for i in range(max_objects):
                if node_mask[i] == 0.0:
                    continue

                px = prediction[i][0]
                py = prediction[i][1]
                pvx = prediction[i][2]
                pvy = prediction[i][3]

                ox = oracle_targets[i][0]
                oy = oracle_targets[i][1]
                ovx = oracle_targets[i][2]
                ovy = oracle_targets[i][3]

                pos_error = abs(px - ox) + abs(py - oy)
                vel_error = abs(pvx - ovx) + abs(pvy - ovy)

                step_error += pos_error + vel_error

            total_error += step_error
            drift_growth += max(0.0, step_error - previous_error)

            previous_error = step_error

        curiosity = total_error + drift_growth * 2.0
        consistency = 1.0 / (1.0 + curiosity)

        return curiosity, consistency

    def score_candidates(self, candidates, max_objects):
        scored = []

        for scene in candidates:
            nodes, edges, targets, node_mask, next_nodes = scene

            curiosity, consistency = self.curiosity_score(
                nodes,
                node_mask,
                max_objects
            )

            scored.append((
                scene,
                curiosity,
                consistency
            ))

        scored = sorted(
            scored,
            key=lambda x: x[1],
            reverse=True
        )

        return scored

    def train_on_scenes(self, selected_scenes, epochs=25):
        dataset = ExplorerDataset(selected_scenes)

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0

            for nodes, edges, targets, node_mask in loader:
                nodes = nodes.to(self.device)
                edges = edges.to(self.device)
                targets = targets.to(self.device)
                node_mask = node_mask.to(self.device)

                self.optimizer.zero_grad()

                prediction, hidden = self.model(
                    nodes,
                    edges,
                    node_mask
                )

                mask = node_mask.unsqueeze(-1)

                loss = ((prediction - targets) ** 2)
                loss = loss * mask
                loss = loss.sum() / mask.sum().clamp(min=1.0)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

                total_loss += float(loss.item())

            if epoch % 5 == 0:
                print(
                    "explorer epoch",
                    epoch,
                    "loss",
                    round(total_loss, 6)
                )

    def explore(self, iterations=5, candidates_per_round=40, top_k=5, max_objects=5):
        for iteration in range(iterations):
            print()
            print("EXPLORATION ITERATION", iteration)
            print("----------------------")

            candidates = self.generate_candidates(
                count=candidates_per_round,
                max_objects=max_objects
            )

            scored = self.score_candidates(
                candidates,
                max_objects
            )

            print()
            print("TOP HIGH-CURIOSITY SCENES")
            print("-------------------------")

            selected_scenes = []

            for index, item in enumerate(scored[:top_k]):
                scene, curiosity, consistency = item

                nodes, edges, targets, node_mask, next_nodes = scene

                selected_scenes.append(scene)

                print(
                    "scene",
                    index,
                    "| objects",
                    int(sum(node_mask)),
                    "| curiosity",
                    round(curiosity, 5),
                    "| consistency",
                    round(consistency, 5)
                )

            self.memory.extend(selected_scenes)

            print()
            print("RETRAINING ON HIGH-CURIOSITY SCENES")
            print("-----------------------------------")

            self.train_on_scenes(
                selected_scenes,
                epochs=25
            )

            print()
            print("POST-TRAIN EVALUATION")
            print("---------------------")

            rescored = self.score_candidates(
                selected_scenes,
                max_objects
            )

            for index, item in enumerate(rescored):
                scene, curiosity, consistency = item

                nodes, edges, targets, node_mask, next_nodes = scene

                print(
                    "scene",
                    index,
                    "| objects",
                    int(sum(node_mask)),
                    "| curiosity",
                    round(curiosity, 5),
                    "| consistency",
                    round(consistency, 5)
                )