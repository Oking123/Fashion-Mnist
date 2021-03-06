import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
import time


# load data
data_mnist = pd.read_csv('~/CNN/train.csv', sep = ',')
test_mnist = pd.read_csv('~/CNN/testX.csv', sep = ',')
mnist_train_data = data_mnist.iloc[:,2:]
mnist_train_label = data_mnist['Label']
mnist_test_data = test_mnist.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(mnist_train_data, mnist_train_label, test_size = 0.166666, random_state = 42)

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train - 0.5, x_test - 0.5
x_train, x_test = x_train * 2, x_test * 2
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model=models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", 
          input_shape=x_train.shape[1:], activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256))
model.add(layers.LeakyReLU())
model.add(layers.Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
                                    optimizer=Adam(learning_rate=0.001),
                                    metrics=['accuracy','mse','mae'])
model.summary()

epochs_number=50

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs_number,batch_size=500)

result1 = model.predict_classes(x_test)
file4 = open("predict.csv",mode='w')
file4.writelines("Id,Label\n")
for i in range(10000):
	file4.writelines("{},{}\n".format(i,result1[i]))
file4.close()

file1 = open("model3loss.csv",mode='w')
file1.writelines("epochs,loss,acc,val_loss,val_acc\n")
for i in range(epochs_number):
	file1.writelines("{},{},{},{},{}\n".format(i+1,history.history['loss'][i],history.history['acc'][i],history.history['val_loss'][i],history.history['val_acc'][i]))
file1.close()
# get test.csv
mnist_test_data = (mnist_test_data/255.0 - 0.5) * 2
mnist_test_data = mnist_test_data.values
mnist_test_data = mnist_test_data.reshape(-1,28,28,1)
result = model.predict_classes(mnist_test_data)
file2 = open("model3result.csv",mode='w')
file2.writelines("Id,Label\n")
for i in range(10000):
	file2.writelines("{},{}\n".format(i,result[i]))
file2.close()

file3 = open("Other_matrix.csv",mode='w')
file3.writelines("epochs,mse,mae,mape\n")
for i in range(epochs_number):
	file3.writelines("{},{},{},{},{}\n".format(i+1,
		history.history['mean_squared_error'][i],
		history.history['mean_absolute_error'][i],
		history.history['val_mean_squared_error'][i],
		history.history['val_mean_absolute_error'][i]))
file3.close()
