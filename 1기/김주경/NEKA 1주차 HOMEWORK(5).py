import torch

def addition_tensors (a, b):
    c = a + b
    return c

def subtraction_tensors (a, b):
    c = a - b
    return c

def multiply_tensors (a, b):
    c = a * b
    return c

tensor1 = torch.tensor([[5, -2], [3, 1]])
tensor2 = torch.tensor([[2, 3], [1, -4]])

result_addition = addition_tensors(tensor1, tensor2)
result_subtraction = subtraction_tensors(tensor1, tensor2)
result_multiply = multiply_tensors(tensor1, tensor2)
print(result_addition, '\n', result_subtraction, '\n', result_multiply)