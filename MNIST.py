print("import dependencies...")
import keras
import numpy
import pandas


print("gather data...")
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("preprocessing...")
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
x_test = x_test/255
x_train = x_train/255
y_test = keras.utils.np_utils.to_categorical(y_test)
y_train = keras.utils.np_utils.to_categorical(y_train)
    
    
print("Compile model...")
model = keras.models.Sequential()
model.add(keras.layers.Dense(784, activation="sigmoid", input_shape=(784,1)))
model.add(keras.layers.Dense(int(784/4), activation='sigmoid'))
model.add(keras.layers.Dense(int(784/16), activation='sigmoid'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


print("Train model")
model.fit(x_train, y_train, epochs=5, batch_size=1000, validation_data(x_test, y_test))




