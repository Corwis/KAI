import random
import torch
from torch.utils.data import Dataset, DataLoader

from data import generate_transition
from model import KaiDynamicsV04

class ExtraScenarioDataset(Dataset):
    def __init__(self, scenarios, repeats=200):
        self.samples = []

        for obj, relation, support, score in scenarios:
            for _ in range(repeats):
                node_a, edge, node_b, target = generate_transition(obj, support, relation)

                self.samples.append((
                    torch.tensor(node_a, dtype=torch.float32),
                    torch.tensor(edge, dtype=torch.float32),
                    torch.tensor(node_b, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class KaiExplorer:
    def __init__(self, model, verifier, optimizer, loss_fn):
        self.model = model
        self.verifier = verifier
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.memory = []

    def generate_candidates(self, count=30):
        objects = ["apple", "stone", "ball", "balloon", "ice"]
        relations = ["on", "in_air"]
        supports = ["table", "floor", "sphere", "small_sphere", "wide_plate"]

        candidates = []

        for _ in range(count):
            obj = random.choice(objects)
            relation = random.choice(relations)
            support = random.choice(supports)

            candidates.append((obj, relation, support))

        return candidates

    def score_candidates(self, candidates):
        scored = []

        for obj, relation, support in candidates:
            result = self.verifier(obj, relation, support, steps=50)
            scored.append((
                obj,
                relation,
                support,
                result["curiosity_score"],
                result["consistency_score"]
            ))

        scored = sorted(scored, key=lambda x: x[3], reverse=True)

        return scored

    def train_on_scenarios(self, scenarios, epochs=80):
        dataset = ExtraScenarioDataset(scenarios, repeats=150)

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0

            for node_a, edge, node_b, target in loader:
                self.optimizer.zero_grad()

                prediction, a_embedding, b_embedding = self.model(node_a, edge, node_b)

                loss = self.loss_fn(prediction, target)

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                print("  explorer epoch", epoch, "loss", round(total_loss, 6))

    def explore(self, iterations=5, candidates_per_round=40, top_k=5):
        for iteration in range(iterations):
            print()
            print("EXPLORATION ITERATION", iteration)
            print("----------------------")

            candidates = self.generate_candidates(candidates_per_round)
            scored = self.score_candidates(candidates)

            print()
            print("TOP CURIOSITY BEFORE TRAINING")
            print("-----------------------------")

            for item in scored[:top_k]:
                obj, relation, support, curiosity, consistency = item

                print(
                    obj,
                    relation,
                    support,
                    "| curiosity =",
                    round(curiosity, 4),
                    "| consistency =",
                    round(consistency, 4)
                )

            selected = [
                (obj, relation, support, curiosity)
                for obj, relation, support, curiosity, consistency in scored[:top_k]
            ]

            self.memory.extend(selected)

            print()
            print("TRAINING ON HIGH-CURIOSITY SCENARIOS")
            print("------------------------------------")

            self.train_on_scenarios(selected)

            rescored = self.score_candidates([
                (obj, relation, support)
                for obj, relation, support, curiosity in selected
            ])

            print()
            print("AFTER TRAINING")
            print("--------------")

            for item in rescored:
                obj, relation, support, curiosity, consistency = item

                print(
                    obj,
                    relation,
                    support,
                    "| curiosity =",
                    round(curiosity, 4),
                    "| consistency =",
                    round(consistency, 4)
                )