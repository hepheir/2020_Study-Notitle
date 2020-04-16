#-*- coding:utf-8 -*-
# ------------------------------------------------
# https://stackoverflow.com/questions/55142951/
# This code includes tf.Session(). Therefore, make v1 features usable.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# ------------------------------------------------
# https://stackoverflow.com/questions/47068709/
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------------------------------

import tensorflow as tf

h = tf.constant("Hello")
w = tf.constant(" World!")
hw = h + w

with tf.Session() as sess:
    ans = sess.run(hw)

print(ans)
