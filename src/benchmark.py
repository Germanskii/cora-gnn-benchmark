"""Compare GCN, GAT and Graph Transformer for transductive node classification on Cora."""
import argparse
import copy, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import f1_score

SEEDS = [42, 43, 44]
EPOCHS = 200
PATIENCE = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT = Path('results/gnn')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphTransformer(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.bn = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_evaluate(model, data, seed, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    best_val, best_state, stale = (-1.0, None, 0)
    history = []
    sync()
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_accuracy': val})
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    sync()
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    yt = data.y[data.test_mask].cpu().numpy()
    yp = pred[data.test_mask].cpu().numpy()
    torch.save({'state_dict': {k: v.cpu() for k, v in best_state.items()}, 'seed': seed, 'model': name}, OUTPUT / f'{name}_{seed}.pt')
    pd.DataFrame(history).to_csv(OUTPUT / f'{name}_{seed}_history.csv', index=False)
    return {'model': name, 'seed': seed, 'val_accuracy': best_val, 'test_accuracy': float((yt == yp).mean()), 'test_f1_macro': f1_score(yt, yp, average='macro', zero_division=0), 'parameters': sum((p.numel() for p in model.parameters())), 'train_seconds': elapsed, 'epochs': len(history)}
def save_figure():
    figure = plt.gcf()
    count = len(list(OUTPUT.glob('figure_*.png'))) + 1
    figure.savefig(OUTPUT / f'figure_{count:02d}.png', dpi=150, bbox_inches='tight')
    plt.close(figure)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare GCN, GAT and Graph Transformer for transductive node classification on Cora.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--data-root', type=Path, default=Path('data/Cora'))
    parser.add_argument('--output', type=Path, default=Path('results/gnn'))
    args = parser.parse_args(argv)
    global EPOCHS, PATIENCE, SEEDS, OUTPUT
    EPOCHS, PATIENCE, SEEDS, OUTPUT = args.epochs, args.patience, args.seeds, args.output
    if EPOCHS < 1 or PATIENCE < 1: parser.error('epochs and patience must be positive')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = Planetoid(root=str(args.data_root), name='Cora')
    data = dataset[0].to(DEVICE)
    for a, b in [('train_mask','val_mask'), ('train_mask','test_mask'), ('val_mask','test_mask')]:
        assert not (getattr(data,a) & getattr(data,b)).any()
    rows = []
    for seed in SEEDS:
        for name, cls, hidden in [('GCN',GCN,64), ('GAT',GAT,8), ('GraphTransformer',GraphTransformer,16)]:
            seed_all(seed)
            model = cls(dataset.num_features, hidden, dataset.num_classes).to(DEVICE)
            rows.append(train_evaluate(model, data, seed, name))
            print(rows[-1])
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT/'metrics.csv', index=False)
    summary = results.groupby('model')[['test_accuracy','test_f1_macro','train_seconds']].agg(['mean','std'])
    print(summary.to_string())
    means = results.groupby('model')['test_accuracy'].mean()
    stds = results.groupby('model')['test_accuracy'].std().fillna(0)
    means.plot.bar(yerr=stds, ylabel='Test accuracy', ylim=(0,1), rot=0)
    plt.tight_layout(); save_figure()


if __name__ == '__main__':
    main()
