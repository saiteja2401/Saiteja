import joblib

#Load a model
model = joblib.load('diabetic_79.pkl')

# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
result = model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])

if result[0] == 0:
    print('Not a diabetic')
else:
    print('Diabetic')
