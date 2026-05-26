import random
import math
import torch
from torch.utils.data import Dataset

OBJECT_TYPES = {
    "apple": [0.20, 0.35, 0.20, 0.85, 0.00, 0.10],
    "stone": [0.85, 0.55, 0.15, 1.00, 0.00, 0.11],
    "ball": [0.30, 1.00, 0.05, 0.75, 0.00, 0.13],
    "balloon": [0.05, 0.85, 0.02, 0.25, 1.00, 0.15],
    "ice": [0.18, 0.45, 0.03, 0.90, 0.00, 0.10],
    "cube": [0.45, 0.05, 0.85, 0.95, 0.00, 0.12],
    "wood": [0.35, 0.15, 0.75, 0.70, 0.00, 0.14]
}

TRAIN_OBJECTS = ["apple", "stone", "ball", "balloon", "cube", "wood"]
TEST_OBJECTS = ["ice"]

NODE_DIM = 14
EDGE_DIM = 12
TARGET_DIM = 5

WORLD_MIN_X = -1.5
WORLD_MAX_X = 1.5
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 2.5

def clamp(value, low, high):
    return max(low, min(high, value))

def object_features(name):
    return OBJECT_TYPES[name]

def make_node(name, x, y, vx, vy):
    base = object_features(name)
    return [
        base[0],
        base[1],
        base[2],
        base[3],
        base[4],
        base[5],
        1.0,
        x,
        y,
        vx,
        vy,
        0.0,
        0.0,
        1.0
    ]

def empty_node():
    return [0.0 for _ in range(NODE_DIM)]

def distance(a, b):
    dx = b[7] - a[7]
    dy = b[8] - a[8]
    return math.sqrt(dx * dx + dy * dy) + 1e-6

def build_edge(a, b):
    dx = b[7] - a[7]
    dy = b[8] - a[8]
    dist = math.sqrt(dx * dx + dy * dy) + 1e-6

    vx_rel = b[9] - a[9]
    vy_rel = b[10] - a[10]

    radius_sum = a[5] + b[5]
    overlap = max(0.0, radius_sum - dist)
    touching = 1.0 if overlap > 0.0 else 0.0

    nx = dx / dist
    ny = dy / dist

    a_above_b = 1.0 if a[8] > b[8] else 0.0
    b_above_a = 1.0 if b[8] > a[8] else 0.0

    support_like = 1.0 if abs(dx) < radius_sum * 0.9 and a[8] > b[8] else 0.0

    return [
        dx,
        dy,
        dist,
        nx,
        ny,
        vx_rel,
        vy_rel,
        overlap,
        touching,
        support_like,
        a_above_b,
        b_above_a
    ]

def build_edges(nodes, node_mask, max_objects):
    edges = []

    for i in range(max_objects):
        row = []

        for j in range(max_objects):
            if i == j or node_mask[i] == 0.0 or node_mask[j] == 0.0:
                row.append([0.0 for _ in range(EDGE_DIM)])
            else:
                row.append(build_edge(nodes[i], nodes[j]))

        edges.append(row)

    return edges

def apply_physics_step(nodes, node_mask, max_objects):
    active_count = int(sum(node_mask))
    next_nodes = [node[:] for node in nodes]
    targets = []

    forces = []

    for i in range(max_objects):
        if node_mask[i] == 0.0:
            forces.append([0.0, 0.0])
            continue

        node = nodes[i]

        mass = max(0.03, node[0])
        buoyancy = node[4]
        movable = node[6]

        if movable < 0.5:
            forces.append([0.0, 0.0])
            continue

        gravity = (1.0 - buoyancy) * -0.035
        lift = buoyancy * 0.030

        fx = 0.0
        fy = gravity + lift

        for j in range(active_count):
            if i == j:
                continue

            other = nodes[j]

            dx = node[7] - other[7]
            dy = node[8] - other[8]
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            radius_sum = node[5] + other[5]
            overlap = radius_sum - dist

            if overlap > 0.0:
                nx = dx / dist
                ny = dy / dist

                rigid_mix = (node[3] + other[3]) * 0.5
                push = overlap * (0.16 + rigid_mix * 0.10)

                fx += nx * push / mass
                fy += ny * push / mass

                if node[8] > other[8] and abs(dx) < radius_sum * 0.75:
                    fy += overlap * 0.20
                    next_nodes[i][11] = 1.0

                friction_mix = (node[2] + other[2]) * 0.5
                fx -= node[9] * friction_mix * 0.020

        if node[8] - node[5] <= WORLD_MIN_Y + 0.001:
            next_nodes[i][11] = 1.0
            fy += max(0.0, -node[10]) * 0.40
            fx -= node[9] * node[2] * 0.05

        forces.append([fx, fy])

    for i in range(max_objects):
        if node_mask[i] == 0.0:
            targets.append([0.0 for _ in range(TARGET_DIM)])
            continue

        node = nodes[i]
        updated = next_nodes[i]

        movable = node[6]
        radius = node[5]

        if movable < 0.5:
            targets.append([node[7], node[8], 0.0, 0.0, 1.0])
            continue

        vx = clamp(node[9] + forces[i][0], -0.45, 0.45)
        vy = clamp(node[10] + forces[i][1], -0.55, 0.55)

        x = clamp(node[7] + vx, WORLD_MIN_X, WORLD_MAX_X)
        y = clamp(node[8] + vy, radius, WORLD_MAX_Y)

        contact = updated[11]

        if y <= radius + 0.001:
            y = radius
            vy = max(0.0, vy)
            vx *= 1.0 - node[2] * 0.1
            contact = 1.0

        updated[7] = x
        updated[8] = y
        updated[9] = vx
        updated[10] = vy
        updated[11] = contact
        updated[12] = math.sqrt(vx * vx + vy * vy)
        updated[13] = 1.0

        next_nodes[i] = updated
        targets.append([x, y, vx, vy, contact])

    return next_nodes, targets

def generate_scene(min_objects=3, max_objects=5, use_test_object=False):
    pool = TEST_OBJECTS if use_test_object else TRAIN_OBJECTS
    count = random.randint(min_objects, max_objects)

    nodes = []

    for i in range(count):
        name = random.choice(pool)

        x = random.uniform(-0.85, 0.85)
        y = random.uniform(0.35, 1.75)
        vx = random.uniform(-0.04, 0.04)
        vy = random.uniform(-0.04, 0.04)

        if len(nodes) > 0 and random.random() < 0.45:
            anchor = random.choice(nodes)
            x = anchor[7] + random.uniform(-0.16, 0.16)
            y = anchor[8] + random.uniform(-0.12, 0.18)

        node = make_node(name, x, y, vx, vy)
        nodes.append(node)

    node_mask = [1.0 for _ in range(count)]

    while len(nodes) < max_objects:
        nodes.append(empty_node())
        node_mask.append(0.0)

    edges = build_edges(nodes, node_mask, max_objects)
    next_nodes, targets = apply_physics_step(nodes, node_mask, max_objects)

    return nodes, edges, targets, node_mask, next_nodes

class KaiMultiObjectDataset(Dataset):
    def __init__(self, size=8000, min_objects=3, max_objects=5, use_test_object=False):
        self.samples = []

        for _ in range(size):
            nodes, edges, targets, node_mask, next_nodes = generate_scene(
                min_objects=min_objects,
                max_objects=max_objects,
                use_test_object=use_test_object
            )

            self.samples.append((
                torch.tensor(nodes, dtype=torch.float32),
                torch.tensor(edges, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(node_mask, dtype=torch.float32),
                torch.tensor(next_nodes, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]