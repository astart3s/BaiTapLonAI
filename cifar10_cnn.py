import numpy as np
import os
import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical



(xtrain, ytrain ), (xtest,ytest) = cifar10.load_data()
# bien doi ytrain -> one hot coding
classes = ['may bay', 'automobile', 'bird', 'cat', 'bear', 'dog', 'frog', 'horse','ship','truck']
# for i in range(10):
#     plt.subplot(2, 5,i+1)
#     plt.imshow(xtrain[i])
#     plt.title(classes[ytrain[i][0]])
#     plt.axis('off')
# plt.show()
xtrain = xtrain/255
xtest = xtest/255
ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

# model_training = models.Sequential([
#     layers.Flatten(input_shape= (32,32,3)), # 32x32, 3 layers
#     layers.Dense(3000,activation= 'relu'),  #fully connective
#     layers.Dense(1000,activation= 'relu'),
#     layers.Dense(10,activation='softmax'),
# ])
# # model_training.summary()
# model_training.compile(optimizer= 'SGD', # update parameter
#                        loss='categorical_crossentropy', #tinh mat mat
#                        metrics='accuracy') # tinh do chinh xac
# model_training.fit(xtrain,ytrain, epochs=10) # data train/label/loop
#
# model_training.save('model cifar10' )
models = models.load_model('model cifar10')
pre = models.predict(xtest[94].reshape((-1,32,32,3))) #return probabilitiy
print (pre)
print (np.argmax(pre)) # return position which is the most high probility
print (classes[np.argmax(pre)])

plt.imshow(xtest[94
                ])
plt.show()