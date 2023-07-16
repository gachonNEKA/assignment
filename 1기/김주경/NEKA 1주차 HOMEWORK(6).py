import torch

def angle90_tensors (a):
    spin = torch.rot90(a, k=1, dims=(0, 1))
    return spin

def angle180_tensors (a):
    spin = torch.rot90(a, k=2, dims=(0, 1))
    return spin

def angle270_tensors (a):
    spin = torch.rot90(a, k=3, dims=(0, 1))
    return spin

def flip_tensors (a):
    flip = torch.flip(a, [0])
    return flip

tensor1 = torch.tensor([[9, 13, 5, 2], [1, 11, 7, 6], [3, 7, 4, 1], [6, 0, 7, 10]])

result_angle90 = angle90_tensors(tensor1)
result_angle180 = angle180_tensors(tensor1)
result_angle270 = angle270_tensors(tensor1)
result_flip = flip_tensors(tensor1)
print(result_angle90, '\n', result_angle180, '\n', result_angle270, '\n', result_flip)
