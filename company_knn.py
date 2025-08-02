import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:/Users/MITS/PycharmProjects/PythonProject2/Company_data.csv')
print(df.head())
print(df.isnull().sum())

feature_value = df.drop('Sales',axis=1)
target_value = df['Sales']


ob = StandardScaler()
scaled_features = ob.fit_transform(feature_value)

x_train,x_test,y_train,y_test = train_test_split(feature_value,target_value,test_size=0.2,random_state=42)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

r2s = r2_score(y_test,y_pred)
print('Performance = ', round(r2s*100,2),'%')

mes = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_pred,y_test)

print('Mean squared Error:',mes)
print('Root Mean Squared Error = ',rmse)
print('Mean Absolute Error = ',mae)