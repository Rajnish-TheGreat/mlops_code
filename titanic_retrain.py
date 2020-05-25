from keras.models import load_model
import os

models = load_model('titanic.h5')# Let's print our layers 
print("Number of layers in the base model: ", len(models.layers))
lay = len(models.layers)

for layers in models.layers[:lay-1]:
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


