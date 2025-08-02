import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Activity 1
df = pd.read_csv("C:/Users/MITS/PycharmProjects/PythonProject2/social_network_ads.csv")
print(df.head())
print(df.isnull().sum())

# Activity 2
feature_df = df.drop(['User ID','Purchased'],axis = 1)
# print(feature_df)
target_df =  df['Purchased']

print("Feature df info \n",feature_df.info())

print("Target df info \n",target_df.info())

x1 = pd.get_dummies(feature_df)
print(x1.head())
x_train,x_test,y_train,y_test = train_test_split(x1,target_df,test_size=0.2,random_state=42)


print("shape of x train      : ",x_train.shape)
print("shape of y train      : ",y_train.shape)
print("shape of x test       : ",x_test.shape)
print("shape of y test       : ",y_test.shape)

# Activity 4
knn  = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

# Activity 5
y_pred = knn.predict(x_test)
r = accuracy_score(y_pred,y_test)
print("Performance = ",round(r*100,2),"%")

print(classification_report(y_test,y_pred))