import torch

def multiply_tensors(a, b): # multiply_tensors라는 함수를 정의하며 함수는 a와 b라는 두 개의 텐서를 인자로 받는다.
    c = a * b

    return c

tensor1 = torch.tensor([2, 4, 6])
tensor2 = torch.tensor([1, 3, 5])

result = multiply_tensors(tensor1, tensor2)
print(result)