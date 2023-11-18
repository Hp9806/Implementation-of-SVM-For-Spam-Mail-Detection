# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by:S Harish Kumar.
RegisterNumber:  212221230104.

```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/Adithya-Siddam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427248/3701f842-2850-4ac5-ac18-9f54f6d6d355)

![image](https://github.com/Adithya-Siddam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427248/a01280e4-7f6e-4997-9a7a-37f07b352a97)

![image](https://github.com/Adithya-Siddam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427248/987870d3-0349-4e48-ba6a-60735cf2346e)

![image](https://github.com/Adithya-Siddam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427248/3dcee172-211a-4f24-82ce-d6dc5b7d4094)

![image](https://github.com/Adithya-Siddam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427248/51b7dca7-4998-488a-a43d-1ce09ba91d79)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
