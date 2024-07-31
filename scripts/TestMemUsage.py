import torch
import torch.nn as nn
import torch.cuda as cuda

# Function to print the current memory usage
def print_memory_usage():
    allocated = cuda.memory_allocated() / (1024 ** 2)
    cached = cuda.memory_reserved() / (1024 ** 2)
    print(f"Memory allocated: {allocated:.2f} MB")
    print(f"Memory cached: {cached:.2f} MB")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple 1D convolutional layer
conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1).to(device)

# Print initial memory usage
print("Initial memory usage:")
print_memory_usage()

# Create a random input tensor of size (32, 1, 32768)
input_tensor = torch.randn(32, 1, 32768).to(device)

# Print memory usage after creating input tensor
print("\nMemory usage after creating input tensor:")
print_memory_usage()

# Pass the input tensor through the convolutional layer
output_tensor = conv1d(input_tensor)

# Print memory usage after passing the tensor through the convolutional layer
print("\nMemory usage after passing input tensor through Conv1D layer:")
print_memory_usage()

# Optionally, perform a backward pass to test memory usage during training
loss = output_tensor.sum()
loss.backward()

# Print memory usage after backward pass
print("\nMemory usage after backward pass:")
print_memory_usage()
