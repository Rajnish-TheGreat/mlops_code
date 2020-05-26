import pandas as pd
import random
import os
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
from keras.models import Model


from keras.models import load_model
import os

models = load_model('titanic.h5')
print("Number of layers in the base model: ", len(models.layers))
lay = len(models.layers)

for layers in models.layers[:lay-2]:
    layers.trainable=False

import random
def add_layer(models_layers,num_class):
    top_model=models_layers.output
    top_model=Dense(random.randint(20,256),activation='relu')(top_model)
    top_model=Dense(num_class,activation='softmax')(top_model)
    
    return top_model
num_class=2

Fc_Head=add_layer(models,num_class)
model=Model(inputs=models.input,outputs=Fc_Head)

model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )

accuracy = model.fit(X,y_cat, epochs=10)
model.save('titanic.h5')

if accuracy.history['accuracy'][-1:][0] < 0.8 :
    os.system("curl --user admin:redhat http://192.168.56.101:8080/job/mlops5/build?token=mlops")


