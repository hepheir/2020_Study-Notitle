# https://www.youtube.com/watch?v=vFQK6daIVr8

import tensorflow as tf

tensor_a = tf.constant(2)
tensor_b = tf.constant(3)
tensor_c = tf.constant(5)

tensor_d = tf.add(tensor_a, tensor_b)
tensor_e = tf.multiply(tensor_d, tensor_c)
print(tensor_e.numpy())

tensor_d = tensor_a + tensor_b
tensor_e = tensor_d * tensor_c
print(tensor_e.numpy())

tensor_A = tf.constant([[1, 2], [3, 4]])
tensor_B = tf.constant([[2, 0], [0, 2]])
tensor_C = tf.matmul(tensor_A, tensor_B)
print(tensor_C.numpy())