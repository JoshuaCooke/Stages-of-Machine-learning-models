import math
import keras
import numpy
from matplotlib import pyplot as plt

lookback = 5
degrees = list(range(360*4))
sine = [math.sin(math.radians(i)) for i in degrees]
x,y =[],[]
for i in range(lookback,len(sine)):
    x.append(sine[i-lookback:i])
    y.append(sine[i])

#reshape
x = numpy.array(x)
x = x.reshape(-1,lookback,1)

# compile model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=32, activation='tanh', input_shape=(lookback,1)))
model.add(keras.layers.Dense(units=8, activation = 'tanh'))
model.add(keras.layers.Dense(units=1, activation = 'tanh'))

model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mse','accuracy'])

history = model.fit(x,y, batch_size=10, epochs=100, validation_split=0.2)

predictions = model.predict(x)
print(len(predictions), len(y))

plt.plot(range(len(predictions)),predictions)
plt.plot(range(len(y)),y)
plt.show()
