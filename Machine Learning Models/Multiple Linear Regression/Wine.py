import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv(r"C:\Users\Sarthak\Downloads\winequality.csv")
x = data.iloc[:,data.columns!="quality"]
y = data.iloc[:,11]
reg = LinearRegression().fit(x,y)
# Find Most Optimal coefficient of all attributes
coeff_df = pd.DataFrame(reg.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)
y_predict = reg.predict(x)
print(r2_score(y,y_predict))