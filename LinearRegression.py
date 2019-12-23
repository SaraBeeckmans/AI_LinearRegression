import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


pd.plotting.register_matplotlib_converters()
file = "train.csv"
data = pd.read_csv(file, sep=";")
cols = data.columns

print(data.head())

plt.title('Train.csv')
plt.xlabel('X')
plt.ylabel('Y')
sns.scatterplot(x=data['X'], y=data['Y'])
sns.regplot(x=data['X'], y=data['Y'])

X = data['X'].values.reshape(-1,1)
Y = data['Y'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X, Y)

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
plt.show(block=True)