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
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(5,activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#plot the model
from tensorflow.keras.utils import to_categorical, plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# get test.csv
mnist_test_data = mnist_test_data/255.0
mnist_test_data = mnist_test_data.values
mnist_test_data = mnist_test_data.reshape(-1,28,28,1)
result = model.predict_classes(mnist_test_data)
