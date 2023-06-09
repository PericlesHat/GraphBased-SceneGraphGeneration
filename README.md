# Graph-based Scene Graph Generation
This project is a set of graph structure-based scene graph generation. All codes based on PyTorch.

### Models
- **GCN**: Simple Graph Convolution Network. Using 2 GCN layers to update objects and relations separately.
- **GraphSAGE**: Simple Graph Sample and AggregatE model. Using 2 GraphSAGE layers to update objects and relations separately.
- **HGNN**: Heterogeneous GNN can handle Object-Object (OO) and Object-Relation-Object (ORO) graphs and uses a custom heterogeneous graph neural network (GNN) for learning and prediction
- **TripleGCN**: Implemented from Wald et al.'s famous SGPN (CVPR'2020). TripleGCN first generates a <subject, predicate, object> triplet to deal with the features in semantic graphs, and then aggregates the objects & relations features.
- More comming soon...

### Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib


### Usage
To use this framework, you need to design your own `obj_vecs`, `rel_vecs`, and `edge_index` matrices as following.

**obj_vecs**: shape `(O, D)`, where `O` is the number of objects in the scene and `D` is the dimension of each object feature vector

**rel_vecs**: shape `(T, D)`, where `T` is the number of relations in the scene

**edge_index**: shape `(T, 2)`, `edge_index[k] = i, j` indicates a triplet < `object_vecs[i]`, `rel_vecs[k]`, `obj_vecs[j]` >


### Note
The code is still being tested. The correctness and validity of the model cannot be guaranteed.

### License
This project is released under the MIT license.
