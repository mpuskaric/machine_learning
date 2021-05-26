import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# load data with pandas
dataframe = pd.read_csv("C:\\Users\\mpuskaric\\Downloads\\Coursera\\machine-learning-ex1\\machine-learning-ex1\\ex1\\ex1data1.csv")
x_train = dataframe["X"]
y_train = dataframe["Y"]

# converting to numpy
x_train = np.array(x_train)
y_train = np.array(y_train)
length=97

# model
X=tf.compat.v1.placeholder(dtype=tf.float32, shape=length)
Y=tf.compat.v1.placeholder(dtype=tf.float32, shape=length)
m=tf.Variable(np.random.randn())
c=tf.Variable(np.random.randn())

# functions
Ypred=tf.add(tf.multiply(X,m),c)
cost=tf.reduce_sum(tf.pow(Ypred-Y,2))/(2*length)
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
init=tf.compat.v1.global_variables_initializer()
iter=1500

# Tensorflow session
with tf.compat.v1.Session() as session:
    session.run(init)
    for i in range(iter):
        session.run(optimizer,feed_dict={X:x_train,Y:y_train})
        if (i + 1) % 100 == 0:
            cost1=session.run(cost,feed_dict={X:x_train,Y:y_train})
            print("Epoch", (i + 1), ": cost =", cost1, "m =", session.run(m), "c =", session.run(c))
    training_cost = session.run(cost,feed_dict={X:x_train,Y:y_train})
    weight = session.run(m)
    bias = session.run(c)

# Calculating the predictions
y_pred = weight*x_train+bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

# plotting
plt.plot(x_train, y_train, 'ro', label ='Original data')
plt.plot(x_train, y_pred, label ='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()