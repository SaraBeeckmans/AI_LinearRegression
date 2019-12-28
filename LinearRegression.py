import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math



def caluclateSum(col, sum):
    for number in col:
        sum = sum + number

    return sum

    #print(head +"= " + str(sum))


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

X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X, Y)

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)
plt.show(block=True)

Xmean = data["X"].mean()
Ymean = data["Y"].mean()
print("Gemiddelde X = " + str(Xmean))
print("Gemiddelde Y = " + str(Ymean))


data["X-Xmean"] = data["X"]-Xmean
data["Y-Ymean"] = data["Y"]-Ymean
data["X-Xmean x Y-Ymean"] = data["X-Xmean"] * data["Y-Ymean"]
data["(X-Xmean)^2"] = data["X-Xmean"] * data["X-Xmean"]
data["(Y-Ymean)^2"] = data["Y-Ymean"] * data["Y-Ymean"]



print(data.head())
print()

sum = 0
sumfirst = caluclateSum(data["X-Xmean x Y-Ymean"], sum)
sumsec = caluclateSum(data["(X-Xmean)^2"], sum)
sumthird = caluclateSum(data["(Y-Ymean)^2"], sum)

print("X-Xmean x Y-Ymean = " + str(sumfirst))
print("(X-Xmean)^2 = " +str(sumsec))
print("(Y-Ymean)^2 = " + str(sumthird))


print()
print("Calculate R")
r = (sumfirst)/math.sqrt((sumsec) * (sumthird))
print(r)



