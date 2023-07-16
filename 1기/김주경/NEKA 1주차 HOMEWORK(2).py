import numpy as np
import pandas as pd

data = {
    'A': np.random.randint(1, 10, 5),
    'B': np.random.randint(1, 10, 5),
    # 1부터 10 사이의 랜덤한 값으로 5개의 행을 만듬.
}

df = pd.DataFrame(data)

df['C'] = df['A'] + df['B']
print(df, '\n')

df['D'] = df.mean(axis=1) # 열의 평균을 계산
print(df)