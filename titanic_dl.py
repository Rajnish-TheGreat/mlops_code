import pandas as pd
import random
from keras.models import Sequential

dataset  =  pd.read_csv('titanic_train.csv')
def lw(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        elif Pclass == 3:
            return 25
        else:
            return 30
    else:
        return age

dataset['Age'] = dataset[['Age', 'Pclass']].apply(lw , axis=1)
dataset.drop('Cabin', axis=1, inplace=True )

y = dataset['Survived']
y_cat = pd.get_dummies(y)

X = dataset[ ['Pclass','Sex', 'Age', 'SibSp', 'Parch' , 'Embarked' ]]

sex = dataset['Sex']
sex = pd.get_dummies(sex, drop_first=True )

pclass = dataset['Pclass']
pclass = pd.get_dummies(pclass, drop_first=True)

sibsp = dataset['SibSp']
sibsp = pd.get_dummies(sibsp, drop_first=True)

parch = dataset['Parch']
parch = pd.get_dummies(parch, drop_first=True)

embarked = dataset['Embarked']
embarked = pd.get_dummies(embarked, drop_first=True)

age = dataset[ 'Age']

X = pd.concat([age, embarked, parch, sibsp, pclass, sex] ,  axis=1)

num=0

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

model  =  Sequential()
model.add(Dense(units=random.randint(8,256) , input_shape=(18,), 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=32 , 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=16, 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=random.randint(8,64), 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer=RMSprop(learning_rate=0.001),  
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )
accuracy = model.fit(X,y_cat, epochs=15)
accuracy.history['accuracy'][-1:][0]

num = num + 1


while num < 6 :
    model  =  Sequential()
    model.add(Dense(units=random.randint(8,256) , input_shape=(18,), 
                activation='relu', 
                kernel_initializer='he_normal' ))
    model.add(Dense(units=32 , 
                activation='relu', 
                kernel_initializer='he_normal' ))
    model.add(Dense(units=16, 
                activation='relu', 
                kernel_initializer='he_normal' ))
    model.add(Dense(units=random.randint(8,64), 
                activation='relu', 
                kernel_initializer='he_normal' ))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(optimizer=RMSprop(learning_rate=0.001),  
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )
    accuracy = model.fit(X,y_cat, epochs=20)
    accuracy.history['accuracy'][-1:][0] 
    num += 1
    if accuracy.history['accuracy'][-1:][0] > 0.80 :
        model.save('titanic.h5')


