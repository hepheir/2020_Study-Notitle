import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --------------------------------

model = tf.keras.models.Sequential([
    Conv2D(filters=32,
           kernel_size=(3,3),
           padding='same',
           activation='relu',
           input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(filters=32,
           kernel_size=(3,3),
           padding='same',
           activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128,
          activation='relu'),
    Dropout(0.2),
    Dense(10,
          activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------

checkpoint = 'resources/checkpoints/' + __file__.split('/')[-1].replace('.py', '')

# --------------------------------

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

    model.fit(x_train, y_train, epochs=12)
    model.save_weights(checkpoint)

    model.evaluate(x_test, y_test, verbose=2)
