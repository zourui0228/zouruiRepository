import pandas as pd
import numpy as np
from sklearn import linear_model

url = '/home/zr/Documents/data/kaggle_data/k1_dgreen_house_price/'
df_train = pd.read_csv(url + 'train.csv')
df_test = pd.read_csv(url + 'test.csv')
reg = linear_model.LinearRegression()
X = df_train[['OverallQual', 'GrLivArea']]
print(X)
# y=log(y+1), make contribution more concentration, improve about 10% accuracy
y = np.log1p(df_train['SalePrice'].values)
t1 = reg.fit(X=X, y=y)
t2 = reg.coef_
print(reg.coef_)
# ---reg apply on test---
Id_test = df_test.Id.values
X_test = df_test[['OverallQual', 'GrLivArea']]
y_test = reg.predict(X=X_test)
y_test = np.exp(y_test) - 1
# ---solution---
count = 0
for y in y_test:
    if y < 0:
        count += 1
print(count)
y_test = [np.abs(y) for y in y_test]
df_solution = pd.DataFrame({'Id': Id_test, 'SalePrice': y_test})

print(df_solution)
df_solution.to_csv(url + 'submission_linearReg.csv', index=False)
