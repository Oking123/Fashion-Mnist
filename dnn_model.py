import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time


# load data
data_mnist = pd.read_csv('train.csv', sep = ',')
test_mnist = pd.read_csv('testX.csv', sep = ',')
mnist_train_data = data_mnist.iloc[:,2:]
mnist_train_label = data_mnist['Label']
mnist_test_data = test_mnist.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(mnist_train_data, mnist_train_label, test_size = 0.166666, random_state = 42)

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)

model=models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#plot the model
from tensorflow.keras.utils import to_categorical, plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

epochs_number = 50

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs_number,batch_size=500)

file1 = open("loss_dnn.csv",mode='w')
file1.writelines("epochs,loss,acc,val_loss,val_acc\n")
for i in range(epochs_number):
	file1.writelines("{},{},{},{},{}\n".format(i+1,history.history['loss'][i],history.history['acc'][i],history.history['val_loss'][i],history.history['val_acc'][i]))
file1.close()
# get test.csv
mnist_test_data = mnist_test_data/255.0
mnist_test_data = mnist_test_data.values
mnist_test_data = mnist_test_data.reshape(-1,28,28)
result = model.predict_classes(mnist_test_data)
file2 = open("dnn_result.csv",mode='w')
file2.writelines("Id,Label\n")
for i in range(10000):
	file2.writelines("{},{}\n".format(i,result[i]))
file2.close()