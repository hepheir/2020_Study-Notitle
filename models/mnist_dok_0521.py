import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --------------------------------

input_shape = (28,28,1)
checkpoint = 'models/checkpoints/mnist_dok_0521'

# --------------------------------

model = tf.keras.models.Sequential([
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='same',
           activation='relu',
           input_shape=(28, 28,1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32,
           kernel_size=(3, 3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------    

if __name__ == "__main__":
    model.summary()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train / 255.0, tuple([x_train.shape[0]] + list(input_shape)))
    x_test  = np.reshape( x_test / 255.0, tuple([ x_test.shape[0]] + list(input_shape)))


    model.fit(x_train, y_train, epochs=12)
    model.save_weights(checkpoint)

    model.evaluate(x_test, y_test, verbose=2)
