import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.gcn_obj = SAGEConv(in_channels, out_channels)
        self.gcn_rel = SAGEConv(in_channels, out_channels)

    def forward(self, obj_vecs, rel_vecs, edge_index):
        edge_index = edge_index.t()
        # update obj_vecs & rel_vecs
        obj_vecs_out = self.gcn_obj(obj_vecs, edge_index)
        rel_vecs_out = self.gcn_rel(rel_vecs, edge_index)

        return obj_vecs_out, rel_vecs_out

if __name__ == "__main__":
    print("---- TEST GraphSAGE ----")

    in_channels = 512
    out_channels = 512
    model = GraphSAGE(in_channels, out_channels)


    num_objects = 10
    num_relations = 20
    obj_vecs = torch.randn(num_objects, in_channels)
    rel_vecs = torch.randn(num_relations, in_channels)
    edge_index = torch.randint(0, num_objects, (2, num_relations))


    updated_obj_vecs, updated_rel_vecs = model(obj_vecs, rel_vecs, edge_index)

    print("Updated obj_vecs shape:", updated_obj_vecs.shape)
    print("Updated rel_vecs shape:", updated_rel_vecs.shape)
