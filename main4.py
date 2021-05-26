import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# load data with pandas
dataframe = pd.read_csv("C:\\Users\\mpuskaric\\Downloads\\Coursera\\machine-learning-ex1\\machine-learning-ex1\\ex1\\ex1data1.csv")
x_train = dataframe["X"]
y_train = dataframe["Y"]

# converting to numpy
x_train = np.array(x_train).reshape(-1,1)
y_train = np.array(y_train)

# create a model
model = Sequential()
model.add(layers.Dense(1,activation='linear',name='layer_1'))
model.compile(optimizer=SGD(learning_rate=0.01),loss=mse)

# first round of training
model.fit(x_train,y_train,epochs=10)

# updating trainable parameters
params = []
weight = np.array([[1.1929]])
bias = np.array([-3.8946])
params.append(weight)
params.append(bias)
model.layers[0].set_weights(params)

# second round of training
model.fit(x_train,y_train,epochs=1500)
model.summary()
for layer in model.layers:
    print(layer.name)
    print("Weights")
    print("Shape: ", layer.get_weights()[0].shape, '\n', layer.get_weights()[0])
    print("Bias")
    print("Shape: ", layer.get_weights()[1].shape, '\n', layer.get_weights()[1], '\n')
#print(model.trainable_variables)

y_line = model.predict(np.array(x_train))
plt.figure(figsize=(10,8))
plt.plot(x_train,y_line, label ='Fitted line')
plt.plot(x_train,y_train, 'ro', label ='Original data')
plt.legend()
plt.show()
