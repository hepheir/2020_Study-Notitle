import tensorflow as tf

h = tf.constant("Hello")
w = tf.constant(" World!")

hw = h + w

print(hw)
print(hw.numpy())