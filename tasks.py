import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)


def compute_output_size_1d(input_array, kernel_array):
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    out_length = input_length - kernel_length + 1
    return out_length

# Example usage
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))




# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Compute the length of the output array
    length = compute_output_size_1d(input_array, kernel_array)
    conv_array = np.zeros(length, dtype=int)
    
    for i in range(length):
        conv_array[i] = input_array[i:i + len(kernel_array)] @ kernel_array
    
    return conv_array

# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 6])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    in_high = input_matrix.shape[0]
    kern_high = kernel_matrix.shape[0]
    in_wide = input_matrix.shape[1]
    kern_wide = kernel_matrix.shape[1]
    high = in_high - kern_high + 1
    wide = in_wide - kern_wide + 1
    dims = (high, wide)
    return dims


# -----------------------------------------------
input_matrix = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel_matrix = np.array([[1, 0], [0, -1]]) 
print(compute_output_size_2d(input_matrix, kernel_matrix))

# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    dims = compute_output_size_2d(input_matrix, kernel_matrix)
    conv_array = np.zeros(dims, dtype=int)
    for i in range(dims[1]):
      for j in range(dims[0]):
        # For code clarity submat
        submat = input_matrix[j:j + kernel_matrix.shape[0], i:i + kernel_matrix.shape[1]]
        conv_array[j,i] = np.sum(np.multiply(submat, kernel_matrix))
    return conv_array


# -----------------------------------------------
input_matrix = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 10]])
kernel_matrix = np.array([[1, 0], [0, -1]]) 
print(convolute_2d(input_matrix, kernel_matrix))