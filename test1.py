import numpy as np
import pandas as pd

score = np.array([[70, 45, 60], [30, 25, 95], [23, 53, 100]])

df = pd.DataFrame(score, columns=['Math', 'Chemistry', 'Physics'])

print(df)
     