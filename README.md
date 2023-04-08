# Scene Graph Generation Using Heterogeneous GNN
This project is a PyTorch-based scene graph generation algorithm. The algorithm can handle Object-Object (OO) and Object-Relation-Object (ORO) graphs and uses a custom heterogeneous graph neural network (GNN) for learning and prediction.

### Installation
Before running this project, please make sure you have installed the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
You can install these dependencies using the following command:

```
pip install torch numpy matplotlib
```

### Usage
First, you need to place the dataset in the data/ folder. Then, you can run the train.py file for training:

```python
python train.py --in_channels_obj 32 --in_channels_rel 32 --out_channels 32
```

You can change the values of in_channels_obj, in_channels_rel, and out_channels as needed. Additionally, you can also change other parameters, such as learning rate, batch size, etc.

After training, you can run the test.py file for testing:

```python
python test.py --in_channels_obj 32 --in_channels_rel 32 --out_channels 32
```

This file will output the accuracy of the model on the test set and generate a loss graph.

### Note
The code is still being tested. The correctness and validity of the model cannot be guaranteed.

License
This project is released under the MIT license.
