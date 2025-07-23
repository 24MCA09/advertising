import numpy as np
import matplotlib .pyplot as plt
import pandas as pd
import seaborn as sns

# Activity 1: Analysing the Dataset
df = pd.read_csv("C:/Users/MITS/Documents/abhijith/insurance_dataset.csv")
print(df.head())

plt.figure(figsize=(8,6))
plt.title("Insurance Details")
plt.scatter(df['age'],df['charges'])
sns.regplot(x=df['age'],y=df['charges'],data=df,line_kws={"color":'red'})
plt.xlabel("age")
plt.ylabel("charges")
plt.grid()
plt.show()

# Activity 2: Train-Test Split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = df['age']
y = df['charges']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Activity 3: Model Training
x1 = np.reshape(x_train,(-1,1))
y1 = np.reshape(y_train,(-1,1))
print("Reshape of x1 : ",x1.shape)
print("Reshape of y1 : ",y1.shape)

model = LinearRegression()
model.fit(x1,y1)


