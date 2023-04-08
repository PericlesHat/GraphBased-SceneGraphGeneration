# Graph-based Scene Graph Generation
This project is a set of graph structure-based scene graph generation. All codes based on PyTorch.

### Models
- **HGNN**: Heterogeneous GNN can handle Object-Object (OO) and Object-Relation-Object (ORO) graphs and uses a custom heterogeneous graph neural network (GNN) for learning and prediction
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
