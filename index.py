"""回帰"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.datasets import load_boston

datasets = load_boston()
data, target = datasets.data, datasets.target
columns = datasets.feature_names

df = pd.DataFrame(data, columns = columns)
df['Target'] = target

t = df['Target'].values #'Target'の配列
x = df.drop(labels=['Target'], axis=1).values #'Target'以外の配列

from sklearn.model_selection import train_test_split
#データを分割、３割をテストに
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)


"""
Step1 アルゴリズムを宣言（今回は重回帰分析）
"""
from sklearn.linear_model import LinearRegression
# アルゴリズムのインスタンスを生成
model = LinearRegression()

"""
Step2 modelの学習
"""
model.fit(x_train, t_train)
print(model.coef_) #学習後のモデルの値

"""
Step3 検証
"""
# 検証結果 1に近いほど正確
print(f'train score: {model.score(x_train, t_train)}')
print(f'test score: {model.score(x_test, t_test)}')

plt.figure(figsize=(10, 7))
plt.bar(x=columns, height=model.coef_) # ヒストグラム
plt.show()


y = model.predict(x_test)
for index, _ in enumerate(y):
    pprint.pprint(f'pre:{y[index]}')
    pprint.pprint(f'aim:{t_test[index]}')
    print('-------------------------')



