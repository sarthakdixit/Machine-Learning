import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv(r"C:\Users\Sarthak\Downloads\headbrain.csv")
x = data["Head Size(cm^3)"].values
y = data["Brain Weight(grams)"].values
m = len(x)
x = x.reshape(m,1)
reg = LinearRegression().fit(x,y)
y_predict = reg.predict(x)
print(r2_score(y,y_predict))
print(reg.coef_)
print(reg.intercept_)