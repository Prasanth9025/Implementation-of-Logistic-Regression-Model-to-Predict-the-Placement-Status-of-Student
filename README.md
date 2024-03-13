# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and
.duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required
modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PRASANTH U
RegisterNumber: 212222220031
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("\nClassification Report:\n",classification_report1)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
## DATA:

![Screenshot 2024-03-13 123235](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/de79f20a-d9e5-40e1-a075-421f4f73c346)

## ENCODED DATA:

![Screenshot 2024-03-13 123309](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/349f6c11-b5aa-4772-b2bb-a5f74eecac87)

## NULL FUNCTION:

![Screenshot 2024-03-13 123333](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/8e9be58a-0515-4da2-b58e-b23d76c8e42c)

## DATA DUPLICATE:

![Screenshot 2024-03-13 123355](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/d9e80e58-f98a-4ee3-9ec3-9fb59d701380)


## ACCURACY:

![Screenshot 2024-03-13 123424](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/b1519365-414f-42f9-ac14-8d76b5d2e200)

## CONFUSION MATRIX:

![Screenshot 2024-03-13 123448](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/c308de3e-790a-41ea-8c37-db720d6cb975)


## CONFUSION REPORT:

![Screenshot 2024-03-13 123516](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/7b6965c7-b04d-4a49-b919-df42be3b9a65)

## PREDICTION OF LR:

![Screenshot 2024-03-13 123846](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/3ad033eb-2e48-451d-9e8e-c6340fbf3b63)

## GRAPH:

![Screenshot 2024-03-13 123940](https://github.com/Prasanth9025/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343686/342e1886-abea-4b47-9f02-f52d766e13e3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
