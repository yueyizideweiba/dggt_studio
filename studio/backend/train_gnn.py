#!/usr/bin/env python
"""离线训练碰撞预测 GNN。

用法（在 dggt 环境，不需要加载场景）：
    python train_gnn.py --data <dataset.npz> --epochs 200 --out gnn_model.pt
"""
import argparse
import numpy as np

from gnn_model import CollisionGNN, train_model, evaluate, FEAT_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="gnn_model.pt")
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data)
    X = d["X"]
    y_col = d["y_col"]
    y_sev = d["y_sev"]
    print(f"样本数 {X.shape[0]}  特征维 {X.shape[1]}  碰撞率 {float(y_col.mean()):.2f}")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(X.shape[0])
    cut = int(X.shape[0] * args.split)
    tr, te = idx[:cut], idx[cut:]

    model, hist = train_model(X[tr], y_col[tr], y_sev[tr], epochs=args.epochs, lr=args.lr, seed=args.seed)
    ev = evaluate(model, X[te], y_col[te], y_sev[te])

    print("\n=== 测试集 ===")
    print(f"  碰撞准确率 {ev['acc_col']:.3f}  (多数类基线 {ev['majority_col']:.3f})")
    print(f"  严重度准确率 {ev['acc_sev']:.3f}  (多数类基线 {ev['majority_sev']:.3f})")
    print(f"  训练末 loss {hist['loss'][-1]:.4f}  首 loss {hist['loss'][0]:.4f}")

    import torch
    torch.save(model.state_dict(), args.out)
    print(f"  模型已保存 {args.out}")


if __name__ == "__main__":
    main()
