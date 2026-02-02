from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = {
    'study_hrs':[1 , 1.5 , 0.5 , 2.9 , 3 , 2.5 , 2 , 7 ,4.5],
    'class_level':[1 , 5 , 8 , 1 , 12 , 12 , 10 , 12 , 7],
    # 'class_level':['Grade1' , 'Grade5' , 'Grade8' , 'SSC_Prt1' , 'FSC_PreEngg' , 'FSC_PreMedical' , 'SSC_Part2' , 'FSC_PreEngg/FSC_PreEngg' , 'Grade7'],
    'Result':[1 , 1 , 1 , 1 , 0 , 0 , 0 , 1 , 1]
#    'Result':['Pass' , 'Pass' , 'Pass/GradeC' , 'Pass' , 'pass/GradeB' , 'Fail' , 'Fail' , 'Pass/GradeA' , 'Pass/GradeA+']
}
print(data)

data_to_df = pd.DataFrame(data)
print(data_to_df)

# data_frame convert csv_file 
data_to_df.to_csv('Student_performance_pred.csv')

call_csv = pd.read_csv('C:/Users/Muhammad Yahya/Downloads/python/ML With ScikitLearn/Supervised Learning Algorithms/Student_performance_pred.csv' , index_col=[0])
df = call_csv
df

# DATA STORE 2 VARIABLES x , y 
X = df[['study_hrs' , 'class_level']] #indep features/input var / input
y = df['Result'] #depent var / label /target var / output

# create model 
my_model = KNeighborsClassifier(n_neighbors=3)
my_model = my_model.fit(X , y) #fit model
my_model #call model

exam_performance_pred = my_model.predict([[8 , 12]])
print(exam_performance_pred)

new_study_hrs = float(input('Enter your study hours.'))
my_class = int(input('Enter your class level..'))
unknown_student_exam_performance_pred = int(input('Enter your result.'))

if (new_study_hrs >= 7 and my_class == 12):
    if unknown_student_exam_performance_pred == 1:
        print('This is FSC_PreEngg/PreMedical Students..')
        print('Congratualations.')    
    else:
        print('----')    

else:
    print('This is not FSC_PreEngg/PreMedical Students..')  