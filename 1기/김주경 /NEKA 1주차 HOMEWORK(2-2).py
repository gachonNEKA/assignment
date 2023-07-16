import torch

def multiply_tensors(a, b): # multiply_tensors라는 함수를 정의하며 함수는 a와 b라는 두 개의 텐서를 인자로 받는다.
    c = a * b

    return c

tensor1 = torch.tensor([2, 4, 6])
# tensor2 = torch.tensor([1, 3]) 코드의 경우 1행 2열의 텐서를 생성하는 코드이다.
# 위에서 두 개의 텐서를 연산하는 함수를 정의하였다.
# 그리고 tensor1의 경우 1행 3열의 텐서이다.
# 따라서 함수에 호출이 된다면 연산을 진행하는 과정에서 열의 개수(요소의 개수)가 일치하지 않아 오류가 발생한다.
# 아래와 같이 수정해주자.
tensor2 = torch.tensor([1, 3, 5])
# tensor 끼리의 데이터 개수를 맞춰줘야 한다.

result = multiply_tensors(tensor1, tensor2)
print(result)
