import pickle

#Load a model
model = pickle.load(open('diabetic_79.sav', 'rb'))

# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
result = model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])

if result[0] == 0:
    print('Not a diabetic')
else:
    print('Diabetic')
