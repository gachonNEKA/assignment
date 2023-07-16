import numpy as np
import pandas as pd

data = {
    'Name': ['철수', '영희', '진희', '정혁'],
    'Age': [17, 23, 42, 29],
    'City': ['서울', '부산', '인천', '광주']
}

df = pd.DataFrame(data)

df['Salary'] = [5000, 650, 2400, 1000] # 새로운 키를 추가하고자 한다면 데이터의 개수는 기존의 키들과 일치하게 맞추어주자.
print(df)