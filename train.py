import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import KaiDynamicsDataset, generate_transition
from model import KaiDynamicsV04

train_data = KaiDynamicsDataset(size=5000, use_test_object=False)

loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

model = KaiDynamicsV04()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()

for epoch in range(500):
    total_loss = 0

    for node_a, edge, node_b, target, obj, relation, support in loader:
        optimizer.zero_grad()

        prediction, a_embedding, b_embedding = model(node_a, edge, node_b)

        loss = loss_fn(prediction, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print("epoch", epoch, "loss", total_loss)

def predict_case(obj, relation, support):
    node_a, edge, node_b, target = generate_transition(obj, support, relation)

    x_a = torch.tensor([node_a], dtype=torch.float32)
    x_e = torch.tensor([edge], dtype=torch.float32)
    x_b = torch.tensor([node_b], dtype=torch.float32)

    with torch.no_grad():
        prediction, a_embedding, b_embedding = model(x_a, x_e, x_b)

    pred = prediction.numpy()[0]

    print()
    print(obj, relation, support)
    print("target next_y      :", round(target[0], 4))
    print("predicted next_y   :", round(float(pred[0]), 4))
    print("target next_vy     :", round(target[1], 4))
    print("predicted next_vy  :", round(float(pred[1]), 4))
    print("target contact     :", round(target[2], 4))
    print("predicted contact  :", round(float(pred[2]), 4))
    print("a_embedding        :", a_embedding.numpy()[0][:8])

def one_step(obj, relation, support, y, vy, contact, support_area):
    node_a, edge, node_b, target = generate_transition(obj, support, relation)

    node_input = node_a[:6] + [
        y,
        vy,
        contact,
        support_area
    ]

    x_a = torch.tensor([node_input], dtype=torch.float32)
    x_e = torch.tensor([edge], dtype=torch.float32)
    x_b = torch.tensor([node_b], dtype=torch.float32)

    with torch.no_grad():
        prediction, a_embedding, b_embedding = model(x_a, x_e, x_b)

    pred = prediction.numpy()[0]

    return {
        "y": float(pred[0]),
        "vy": float(pred[1]),
        "contact": float(pred[2]),
        "embedding": a_embedding.numpy()[0]
    }

def rollout(obj, relation, support, steps=15, verbose=True):
    node_a, edge, node_b, target = generate_transition(obj, support, relation)

    current_y = node_a[6]
    current_vy = node_a[7]
    current_contact = node_a[8]
    support_area = node_a[9]

    trajectory = []

    if verbose:
        print()
        print("ROLLOUT")
        print("-------")
        print(obj, relation, support)
        print()

    for step in range(steps):
        result = one_step(
            obj,
            relation,
            support,
            current_y,
            current_vy,
            current_contact,
            support_area
        )

        trajectory.append(result)

        if verbose:
            print(
                "step",
                step,
                "| y =", round(result["y"], 4),
                "| vy =", round(result["vy"], 4),
                "| contact =", round(result["contact"], 4)
            )

        current_y = result["y"]
        current_vy = result["vy"]
        current_contact = result["contact"]

    return trajectory

def oracle_step(obj, relation, support, y, vy, contact, support_area):
    node_a, edge, node_b, target = generate_transition(obj, support, relation)

    obj_features = node_a[:6]
    support_features = node_b[:6]

    object_roundness = obj_features[1]
    buoyancy = obj_features[4]

    support_roundness = support_features[1]
    support_flatness = support_features[2]

    if relation == "on":
        real_support_area = max(
            0.05,
            support_flatness * 0.9 + (1.0 - support_roundness) * 0.3
        )

        instability = support_roundness * 0.9 + object_roundness * 0.15 - real_support_area * 0.8
    else:
        real_support_area = 0.0
        instability = 1.0

    gravity_effect = (1.0 - buoyancy) * 0.12
    upward_effect = buoyancy * 0.08

    if relation == "on" and instability < 0.25:
        next_vy = 0.0
        next_y = y
        next_contact = 1.0
    else:
        next_vy = vy - gravity_effect + upward_effect
        next_y = y + next_vy
        next_contact = 0.0

    return {
        "y": next_y,
        "vy": next_vy,
        "contact": next_contact,
        "support_area": real_support_area
    }

def oracle_rollout(obj, relation, support, start_y, start_vy, start_contact, support_area, steps=15):
    y = start_y
    vy = start_vy
    contact = start_contact

    trajectory = []

    for step in range(steps):
        result = oracle_step(
            obj,
            relation,
            support,
            y,
            vy,
            contact,
            support_area
        )

        trajectory.append(result)

        y = result["y"]
        vy = result["vy"]
        contact = result["contact"]

    return trajectory

def verify_rollout(obj, relation, support, steps=30):
    node_a, edge, node_b, target = generate_transition(obj, support, relation)

    start_y = node_a[6]
    start_vy = node_a[7]
    start_contact = node_a[8]
    support_area = node_a[9]

    model_traj = rollout(
        obj,
        relation,
        support,
        steps=steps,
        verbose=False
    )

    oracle_traj = oracle_rollout(
        obj,
        relation,
        support,
        start_y,
        start_vy,
        start_contact,
        support_area,
        steps=steps
    )

    position_errors = []
    velocity_errors = []
    contact_errors = []
    drift_growth = []

    last_total_error = 0.0

    for i in range(steps):
        py = model_traj[i]["y"]
        pvy = model_traj[i]["vy"]
        pc = model_traj[i]["contact"]

        oy = oracle_traj[i]["y"]
        ovy = oracle_traj[i]["vy"]
        oc = oracle_traj[i]["contact"]

        pos_error = abs(py - oy)
        vel_error = abs(pvy - ovy)
        contact_error = abs(pc - oc)

        total_error = pos_error + vel_error + contact_error
        growth = max(0.0, total_error - last_total_error)

        position_errors.append(pos_error)
        velocity_errors.append(vel_error)
        contact_errors.append(contact_error)
        drift_growth.append(growth)

        last_total_error = total_error

    avg_position_error = sum(position_errors) / len(position_errors)
    avg_velocity_error = sum(velocity_errors) / len(velocity_errors)
    avg_contact_error = sum(contact_errors) / len(contact_errors)
    avg_drift_growth = sum(drift_growth) / len(drift_growth)

    final_error = (
        position_errors[-1]
        + velocity_errors[-1]
        + contact_errors[-1]
    )

    curiosity_score = (
        avg_position_error * 1.5
        + avg_velocity_error * 2.0
        + avg_contact_error * 1.0
        + avg_drift_growth * 3.0
        + final_error * 0.5
    )

    consistency_score = 1.0 / (1.0 + curiosity_score)

    return {
        "obj": obj,
        "relation": relation,
        "support": support,
        "avg_position_error": avg_position_error,
        "avg_velocity_error": avg_velocity_error,
        "avg_contact_error": avg_contact_error,
        "avg_drift_growth": avg_drift_growth,
        "final_error": final_error,
        "curiosity_score": curiosity_score,
        "consistency_score": consistency_score
    }

def print_verification(result):
    print()
    print(result["obj"], result["relation"], result["support"])
    print("avg_position_error :", round(result["avg_position_error"], 5))
    print("avg_velocity_error :", round(result["avg_velocity_error"], 5))
    print("avg_contact_error  :", round(result["avg_contact_error"], 5))
    print("avg_drift_growth   :", round(result["avg_drift_growth"], 5))
    print("final_error        :", round(result["final_error"], 5))
    print("curiosity_score    :", round(result["curiosity_score"], 5))
    print("consistency_score  :", round(result["consistency_score"], 5))

def curiosity_scan():
    cases = [
        ("apple", "on", "table"),
        ("apple", "on", "sphere"),
        ("stone", "in_air", "floor"),
        ("balloon", "in_air", "floor"),
        ("ice", "on", "table"),
        ("ice", "on", "sphere"),
        ("ice", "in_air", "floor"),
        ("balloon", "on", "sphere")
    ]

    results = []

    for obj, relation, support in cases:
        result = verify_rollout(
            obj,
            relation,
            support,
            steps=50
        )

        results.append(result)

    results = sorted(
        results,
        key=lambda x: x["curiosity_score"],
        reverse=True
    )

    print()
    print("CURIOSITY SCAN")
    print("--------------")

    for result in results:
        print(
            result["obj"],
            result["relation"],
            result["support"],
            "| curiosity =",
            round(result["curiosity_score"], 5),
            "| consistency =",
            round(result["consistency_score"], 5)
        )

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

        scored = sorted(
            scored,
            key=lambda x: x[3],
            reverse=True
        )

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

    def explore(self, iterations=3, candidates_per_round=30, top_k=5):
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

print()
print("TRAINED OBJECTS")
print("---------------")

predict_case("apple", "on", "table")
predict_case("apple", "on", "sphere")
predict_case("stone", "in_air", "floor")
predict_case("balloon", "in_air", "floor")

print()
print("UNSEEN OBJECT: ICE")
print("------------------")

predict_case("ice", "on", "table")
predict_case("ice", "on", "sphere")
predict_case("ice", "in_air", "floor")

print()
print("ROLLOUT TESTS")
print("-------------")

rollout("apple", "on", "sphere", steps=15)
rollout("apple", "on", "table", steps=15)
rollout("ice", "on", "sphere", steps=15)
rollout("balloon", "in_air", "floor", steps=15)

print()
print("REALITY VERIFIER")
print("----------------")

print_verification(verify_rollout("apple", "on", "table", steps=50))
print_verification(verify_rollout("apple", "on", "sphere", steps=50))
print_verification(verify_rollout("ice", "on", "sphere", steps=50))
print_verification(verify_rollout("balloon", "in_air", "floor", steps=50))

curiosity_scan()

print()
print("KAI EXPLORER V0.7")
print("-----------------")

explorer = KaiExplorer(
    model=model,
    verifier=verify_rollout,
    optimizer=optimizer,
    loss_fn=loss_fn
)

explorer.explore(
    iterations=3,
    candidates_per_round=30,
    top_k=5
)