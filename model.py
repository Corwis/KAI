import torch
import torch.nn as nn

class KaiDynamicsV04(nn.Module):
    def __init__(self, node_dim=10, edge_dim=2, hidden=96):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.message_net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, node_a, edge, node_b):
        a = self.node_encoder(node_a)
        b = self.node_encoder(node_b)
        e = self.edge_encoder(edge)

        message_to_a = self.message_net(torch.cat([a, e, b], dim=-1))
        message_to_b = self.message_net(torch.cat([b, e, a], dim=-1))

        a_updated = self.update_net(torch.cat([a, message_to_a], dim=-1))
        b_updated = self.update_net(torch.cat([b, message_to_b], dim=-1))

        scene = torch.cat([a_updated, b_updated], dim=-1)

        raw = self.dynamics_head(scene)

        next_y = raw[:, 0:1]
        next_vy = raw[:, 1:2]
        next_contact = torch.sigmoid(raw[:, 2:3])

        return torch.cat([next_y, next_vy, next_contact], dim=-1), a_updated, b_updated