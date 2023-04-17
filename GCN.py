import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

"""
GCN demo: update obj & rel separately using 2 GCN Conv layers
"""

class GCN(nn.Module):
    def __init__(self, in_channels):
        super(GCN, self).__init__()
        self.gcn_obj = GCNConv(in_channels, in_channels)
        self.gcn_rel = GCNConv(in_channels, in_channels)

    def forward(self, obj_vecs, rel_vecs, edge_index):
        edge_index = edge_index.t()
        # update obj_vecs & rel_vecs
        obj_vecs_updated = F.relu(self.gcn_obj(obj_vecs, edge_index))
        rel_vecs_updated = F.relu(self.gcn_rel(rel_vecs, edge_index))

        return obj_vecs_updated, rel_vecs_updated


if __name__ == "__main__":
    print("---- TEST GCN ----")

    # demo
    in_channels = 512
    gcn_update = GCN(in_channels)

    O = 10
    T = 15

    obj_vecs = torch.randn(O, in_channels)
    rel_vecs = torch.randn(T, in_channels)
    edge_index = torch.randint(0, O, (T, 2))

    obj_vecs_updated, rel_vecs_updated = gcn_update(obj_vecs, rel_vecs, edge_index)
