import numpy as np

a = np.array([0, 4, 3, 2, 0, 1, 4, 1, 2, 3])
p = np.array([0.4, 0.5, 0.6, 1.5, 1, 2.3, 5.1, 6.7, 1.2, 0.4])

# Get the unique values in `a`
unique_values = np.unique(a)

# Initialize the resulting array `v` with zeros
v = np.zeros_like(unique_values, dtype=float)

# Iterate over the unique values in `a`
for val in unique_values:
    # Get the indices where `a` is equal to the current value
    indices = np.where(a == val)
    
    # Sum the corresponding elements in `p`
    v[val] = np.sum(p[indices])

print(v)

