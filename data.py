import random
import torch
from torch.utils.data import Dataset

object_types = {
    "apple": [0.2, 0.4, 0.1, 0.8, 0.0, 1.0],
    "stone": [0.8, 0.7, 0.1, 1.0, 0.0, 1.0],
    "ball": [0.3, 1.0, 0.0, 0.7, 0.0, 1.0],
    "balloon": [0.05, 0.8, 0.0, 0.2, 1.0, 1.0],
    "ice": [0.15, 0.5, 0.1, 0.9, 0.0, 1.0]
}

support_types = {
    "table": [5.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "floor": [20.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "sphere": [2.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "small_sphere": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "wide_plate": [3.0, 0.0, 1.0, 1.0, 0.0, 0.0]
}

relations = {
    "on": [1.0, 0.0],
    "in_air": [0.0, 1.0]
}

train_objects = ["apple", "stone", "ball", "balloon"]
test_objects = ["ice"]

train_supports = ["table", "floor", "sphere", "small_sphere", "wide_plate"]

def object_vector(name):
    return object_types[name]

def support_vector(name):
    return support_types[name]

def relation_vector(name):
    return relations[name]

def generate_transition(obj, support, relation):
    obj_vec = object_vector(obj)
    sup_vec = support_vector(support)

    mass = obj_vec[0]
    object_roundness = obj_vec[1]
    buoyancy = obj_vec[4]

    support_roundness = sup_vec[1]
    support_flatness = sup_vec[2]

    y = random.uniform(0.8, 1.2)
    vy = random.uniform(-0.02, 0.02)

    if relation == "on":
        contact = 1.0
        support_area = max(0.05, support_flatness * 0.9 + (1.0 - support_roundness) * 0.3)
        instability = support_roundness * 0.9 + object_roundness * 0.15 - support_area * 0.8
    else:
        contact = 0.0
        support_area = 0.0
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

    t0_dynamic = [y, vy, contact, support_area]
    target = [next_y, next_vy, next_contact]

    node_a = obj_vec + t0_dynamic
    node_b = sup_vec + [0.0, 0.0, 1.0, support_area]
    edge = relation_vector(relation)

    return node_a, edge, node_b, target

class KaiDynamicsDataset(Dataset):
    def __init__(self, size=4000, use_test_object=False):
        self.samples = []
        objects = test_objects if use_test_object else train_objects

        for _ in range(size):
            obj = random.choice(objects)
            support = random.choice(train_supports)
            relation = random.choice(["on", "in_air"])

            node_a, edge, node_b, target = generate_transition(obj, support, relation)

            self.samples.append((
                torch.tensor(node_a, dtype=torch.float32),
                torch.tensor(edge, dtype=torch.float32),
                torch.tensor(node_b, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                obj,
                relation,
                support
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]