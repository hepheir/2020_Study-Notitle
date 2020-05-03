import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --------------------------------

input_shape = (28,28)
checkpoint = 'resources/checkpoints/' + __file__.split('/')[-1].replace('.py', '')

# --------------------------------

model = tf.keras.models.Sequential([
            Flatten(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.fit(x_train, y_train, epochs=8)
    model.save_weights(checkpoint)

    model.evaluate(x_test, y_test, verbose=2)