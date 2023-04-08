import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class CustomGNNLayer(MessagePassing):
    def __init__(self, in_channels_obj, in_channels_rel, out_channels):
        super(CustomGNNLayer, self).__init__(aggr='add')
        self.lin_obj = nn.Linear(in_channels_obj, out_channels)
        self.lin_rel = nn.Linear(in_channels_rel, out_channels)

    def forward(self, obj_vecs, rel_vecs, edge_index):
        # Prepare edge attributes based on edge_index
        obj_vecs_i = obj_vecs[edge_index[:, 0]]
        obj_vecs_j = obj_vecs[edge_index[:, 1]]
        rel_vecs_k = rel_vecs[torch.arange(edge_index.size(0))]

        edge_attrs = torch.cat([obj_vecs_i, rel_vecs_k, obj_vecs_j], dim=-1)

        return self.propagate(edge_index.t(), x=obj_vecs, edge_attr=edge_attrs)

    def message(self, x_j, edge_attr):
        obj_vecs_i, rel_vecs_k, obj_vecs_j = torch.split(edge_attr, x_j.size(-1), dim=-1)
        return F.relu(self.lin_obj(obj_vecs_i) + self.lin_rel(rel_vecs_k) + self.lin_obj(obj_vecs_j))

    def update(self, aggr_out):
        return aggr_out

class HeterogeneousGNN(nn.Module):
    def __init__(self, in_channels_obj, in_channels_rel, out_channels):
        super(HeterogeneousGNN, self).__init__()
        self.gnn_layer = CustomGNNLayer(in_channels_obj, in_channels_rel, out_channels)

    def forward(self, obj_vecs, rel_vecs, edge_index):
        x = self.gnn_layer(obj_vecs, rel_vecs, edge_index)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    # Sample usage
    in_channels_obj = 512
    in_channels_rel = 512
    out_channels = 512

    heterogeneous_gnn = HeterogeneousGNN(in_channels_obj, in_channels_rel, out_channels)

    # Prepare obj_vecs, rel_vecs, and edge_index as torch tensors
    # obj_vecs and rel_vecs should have shapes (O, in_channels_obj) and (T, in_channels_rel), respectively
    # edge_index should have shape (E, 2)

    output = heterogeneous_gnn(obj_vecs, rel_vecs, edge_index)
