import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np

call_df = pd.read_csv('C:/Users/Muhammad Yahya/Downloads/python/ML With ScikitLearn/Supervised Learning Algorithms/Projects/Project1_Use_Algo_Linear_Regression/StudentExamScore.csv' , index_col=[0])
print(call_df)

X = call_df[['Hours']] #independent var / input features
y = call_df['Score'] #dependent var / Label

my_model = LinearRegression().fit(X , y)
y_predict = my_model.predict(X)
print(y_predict) #machine predict

# model evaluation 
mae = mean_absolute_error(y , y_predict)
mse = mean_squared_error(y , y_predict)
rmse = np.sqrt(mse)

print(f'MAE = {mae}')
print(f'MSE = {mse}')
print(f'RMSE = {rmse}')

# input from users
new_study_hrs = int(input('Enter your study hours.'))
my_model_predict = my_model.predict([[new_study_hrs]])[0]
print(my_model_predict) #machine predict

if my_model_predict >= 100:         
    print('Your Score Is 99 out of 100 (99% Cleared)')
    print('Congratualations Your Performance Is Very Best') 
else:
    print('----')
