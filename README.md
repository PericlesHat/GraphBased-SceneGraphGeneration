# Graph-based Scene Graph Generation
This project is a set of graph structure-based scene graph generation. All codes based on PyTorch.

### Models
- **HGNN**: Heterogeneous GNN can handle Object-Object (OO) and Object-Relation-Object (ORO) graphs and uses a custom heterogeneous graph neural network (GNN) for learning and prediction
- More comming soon...

### Installation
Before running this project, please make sure you have installed the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib



### Usage
To use this algorithm, you need to design your own `obj_vecs`, `rel_vecs`, and `edge_index` matrices.

`obj_vecs`: shape `(O, D)`, where `O` is the number of objects in the scene and `D` is the dimension of each object feature vector

`rel_vecs`: shape `(T, D)`, where `T` is the number of relations in the scene

`edge_index`: shape `(T, 2)`

Each row in edge_index represents an edge between two objects, and the values in each row represent the indices of the connected objects in the obj_vecs matrix.

### Note
The code is still being tested. The correctness and validity of the model cannot be guaranteed.

### License
This project is released under the MIT license.
