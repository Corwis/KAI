import torch
import torch.nn as nn

class KaiDynamics(nn.Module):
    def __init__(self, node_dim=14, edge_dim=12, hidden=128, message_steps=4, target_dim=5):
        super().__init__()

        self.hidden = hidden
        self.message_steps = message_steps

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        self.message_net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        self.norm = nn.LayerNorm(hidden)

        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, target_dim)
        )

    def forward(self, nodes, edges, node_mask):
        batch_size, object_count, _ = nodes.shape

        h = self.node_encoder(nodes)
        e = self.edge_encoder(edges)

        h = h * node_mask.unsqueeze(-1)

        receiver_mask = node_mask.unsqueeze(2)
        sender_mask = node_mask.unsqueeze(1)
        pair_mask = receiver_mask * sender_mask
        pair_mask = pair_mask.unsqueeze(-1)

        for _ in range(self.message_steps):
            receiver = h.unsqueeze(2).expand(batch_size, object_count, object_count, self.hidden)
            sender = h.unsqueeze(1).expand(batch_size, object_count, object_count, self.hidden)

            message_input = torch.cat([receiver, e, sender], dim=-1)
            messages = self.message_net(message_input)
            messages = messages * pair_mask

            aggregated = messages.sum(dim=2)

            degree = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
            aggregated = aggregated / degree

            update_input = torch.cat([h, aggregated], dim=-1)
            h_next = self.update_net(update_input)

            h = self.norm(h + h_next)
            h = h * node_mask.unsqueeze(-1)

        raw = self.dynamics_head(h)

        next_x = raw[:, :, 0:1]
        next_y = raw[:, :, 1:2]
        next_vx = raw[:, :, 2:3]
        next_vy = raw[:, :, 3:4]
        contact = torch.sigmoid(raw[:, :, 4:5])

        prediction = torch.cat([next_x, next_y, next_vx, next_vy, contact], dim=-1)
        prediction = prediction * node_mask.unsqueeze(-1)

        return prediction, h