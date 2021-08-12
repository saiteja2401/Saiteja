import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names = names)
print(df)
 
X = df.iloc[:, :8]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8, random_state = 101)

#fit the model

model = LogisticRegression()

model.fit(X_train, y_train)

#accuracy
result = model.score(X_test, y_test)
print(result)

#Saving the model and need to use only 2 extensions they are - (pkl, sav)
joblib.dump(model, 'diabetic_79.pkl')













