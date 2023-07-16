import numpy as np
import pandas as pd

data = {
    'Name': ['철수', '영희', '진희', '정혁'],
    'Age': [17, 23, 42, 29],
    'City': ['서울', '부산', '인천', '광주']
}

df = pd.DataFrame(data)
# df['Salary'] = [5000, 650, 2400] 이 코드는 'Salary'라는 새로운 key를 추가하고 3가지의 정수형 value를 추가하는 코드다.
# 그러나 앞의 딕셔너리에서 key와 value를 key 하나 당 value 4개로 매칭해주었다.
# 그러기에 value간의 개수가 일치하지 않아 오류가 발생한다.
# 아래와 같이 수정하자.
df['Salary'] = [5000, 650, 2400, 1000] # 새로운 키를 추가하고자 한다면 데이터의 개수는 기존의 키들과 일치하게 맞추어주자.
print(df)
