"""碰撞预测图神经网络（GNN）—— 独立于后端场景的 torch 模型 + 训练。

输入：一对动态物体的**关系图特征**（节点 + 边 + 物理参数），即 `scene_graph` 产出的
定长特征向量。输出：是否碰撞（二分类）+ 碰撞严重度（4 类）。

特征布局（与 /api/gnn/collect 写入的 .npz 一致）：
  [0:4]   node_a: [speed, log1p(vol), is_vehicle, is_pedestrian]
  [4:8]   node_b: 同上
  [8:18]  edge features（scene_graph 的 10 维关系特征）
  [18:22] 物理参数: [relative_speed, initial_gap, lateral_offset, reaction_delay]

这是"图神经网络"的成对（edge-centric）形式：先用节点 MLP 编码两个物体，
再在图边上融合两节点嵌入 + 边特征 + 参数，输出边级别的碰撞/严重度预测。
后续扩展到多节点场景时，只需在边嵌入之上再加一层消息传递即可。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

NODE_DIM = 4
EDGE_DIM = 10
PARAM_DIM = 4
FEAT_DIM = NODE_DIM * 2 + EDGE_DIM + PARAM_DIM  # 22


class CollisionGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM,
                 param_dim: int = PARAM_DIM, hidden: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.param_dim = param_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim + param_dim, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
        )
        self.coll_head = nn.Linear(hidden, 2)   # 是否碰撞
        self.sev_head = nn.Linear(hidden, 4)    # 严重度 0/1/2/3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nd = self.node_dim
        a = self.node_mlp(x[:, :nd])
        b = self.node_mlp(x[:, nd:2 * nd])
        edge = x[:, 2 * nd:2 * nd + self.edge_dim]
        param = x[:, 2 * nd + self.edge_dim:]
        h = self.edge_mlp(torch.cat([a, b, edge, param], dim=-1))
        return self.coll_head(h), self.sev_head(h)

    def predict(self, x: torch.Tensor):
        logit_col, logit_sev = self.forward(x)
        return logit_col.argmax(-1), logit_sev.argmax(-1)


def train_model(X: np.ndarray, y_col: np.ndarray, y_sev: np.ndarray,
                epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                batch_size: int = 128, seed: int = 0,
                device: Optional[str] = None) -> Tuple[CollisionGNN, dict]:
    """训练并返回模型 + 训练曲线。"""
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    col = torch.tensor(np.asarray(y_col, dtype=np.int64), device=device)
    sev = torch.tensor(np.asarray(y_sev, dtype=np.int64), device=device)

    # 简单归一化：按列 z-score（避免 NaN）
    mean = Xt.mean(0, keepdim=True)
    std = Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mean) / std

    model = CollisionGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    n = Xt.shape[0]
    history = {"loss": [], "acc_col": [], "acc_sev": [], "col_rate": float(col.float().mean())}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            lc, ls = model(Xt[idx])
            loss = ce(lc, col[idx]) + ce(ls, sev[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * idx.shape[0]
        epoch_loss /= max(1, n)

        model.eval()
        with torch.no_grad():
            lc, ls = model(Xt)
            acc_col = float((lc.argmax(-1) == col).float().mean())
            acc_sev = float((ls.argmax(-1) == sev).float().mean())
        history["loss"].append(epoch_loss)
        history["acc_col"].append(acc_col)
        history["acc_sev"].append(acc_sev)
    return model, history


def evaluate(model: CollisionGNN, X: np.ndarray, y_col: np.ndarray,
             y_sev: np.ndarray, device: Optional[str] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    mean = Xt.mean(0, keepdim=True)
    std = Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mean) / std
    col = torch.tensor(np.asarray(y_col, dtype=np.int64), device=device)
    sev = torch.tensor(np.asarray(y_sev, dtype=np.int64), device=device)
    model.eval()
    with torch.no_grad():
        lc, ls = model(Xt)
    return {
        "acc_col": float((lc.argmax(-1) == col).float().mean()),
        "acc_sev": float((ls.argmax(-1) == sev).float().mean()),
        "majority_col": float(max((col == c).float().mean() for c in range(2))),
        "majority_sev": float(max((sev == c).float().mean() for c in range(4))),
        "n": int(Xt.shape[0]),
    }
