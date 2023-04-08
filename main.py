import torch.optim as optim
import matplotlib.pyplot as plt
from HGNN import *


# Training settings
epochs = 100
learning_rate = 0.01

# Sample data
num_obj = 10
num_rel = 20
in_channels_obj = 32
in_channels_rel = 32
out_channels = 32

obj_vecs = torch.randn(num_obj, in_channels_obj)
rel_vecs = torch.randn(num_rel, in_channels_rel)
edge_index = torch.randint(0, num_obj, (num_rel, 2))
print(edge_index.shape)

# Ground truth output for demonstration purpose
obj_vecs_gt = torch.randn(num_obj, out_channels)

# Initialize the model
model = HeterogeneousGNN(in_channels_obj, in_channels_rel, out_channels)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(obj_vecs, rel_vecs, edge_index)
    loss = criterion(output, obj_vecs_gt)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

# Plot the loss graph
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
