import cupy as cp
import numpy as np

def get_vector_input(size, name):
    """Helper function to get vector elements from user or generate random values."""
    print(f"Enter {size} elements for vector {name} (or press Enter to generate random values):")
    user_input = input().strip()
    
    if user_input == "":
        # Generate random values if user skips input
        return cp.random.random(size, dtype=cp.float32)
    else:
        # Parse user input (space-separated values)
        try:
            values = [float(x) for x in user_input.split()]
            if len(values) != size:
                raise ValueError(f"Expected {size} elements, but got {len(values)}")
            return cp.array(values, dtype=cp.float32)
        except ValueError as e:
            print(f"Error: {e}. Generating random values instead.")
            return cp.random.random(size, dtype=cp.float32)

# Get vector size from user
while True:
    try:
        N = int(input("Enter the size of the vectors (e.g., 5 for small, 1000000 for large): "))
        if N <= 0:
            print("Size must be positive!")
            continue
        break
    except ValueError:
        print("Invalid input! Please enter a valid integer.")

# Get vectors (user input or random)
if N <= 10:  # For small vectors, prompt for elements
    d_A = get_vector_input(N, "A")
    d_B = get_vector_input(N, "B")
else:  # For large vectors, use random values by default
    print(f"Vector size {N} is large. Using random values for vectors A and B.")
    d_A = cp.random.random(N, dtype=cp.float32)
    d_B = cp.random.random(N, dtype=cp.float32)

# Perform vector addition on GPU with timing
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
d_C = d_A + d_B
end.record()
end.synchronize()  # Wait for GPU computation to complete
time_ms = cp.cuda.get_elapsed_time(start, end)  # Time in milliseconds

# Copy results to host (CPU) for display and verification
h_A = cp.asnumpy(d_A)
h_B = cp.asnumpy(d_B)
h_C = cp.asnumpy(d_C)

# Display vectors (show first 5 and last 5 elements if large)
print("\nVector A:")
if N <= 10:
    print(h_A)
else:
    print(f"First 5: {h_A[:5]}, Last 5: {h_A[-5:]} (Total {N} elements)")

print("Vector B:")
if N <= 10:
    print(h_B)
else:
    print(f"First 5: {h_B[:5]}, Last 5: {h_B[-5:]} (Total {N} elements)")

print("Result (A + B):")
if N <= 10:
    print(h_C)
else:
    print(f"First 5: {h_C[:5]}, Last 5: {h_C[-5:]} (Total {N} elements)")

# Display timing result
print(f"Vector addition took {time_ms:.3f} milliseconds")

# Verify result
for i in range(N):
    if abs(h_A[i] + h_B[i] - h_C[i]) > 1e-5:
        print(f"Verification failed at index {i}!")
        break
else:
    print("Vector addition completed successfully!")
