import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --------------------------------

model = tf.keras.models.Sequential([
    # FILL
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --------------------------------

checkpoint = 'resources/checkpoints/' + __file__.split('/')[-1].replace('.py', '')

# --------------------------------

if __name__ == "__main__":
    # GET TRAIN

    model.fit(x_train, y_train, epochs=16)
    model.evaluate(x_test, y_test, verbose=2)

    model.save_weights(checkpoint)