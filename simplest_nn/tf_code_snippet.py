import numpy as np

import tensorflow as tf;

def mask_s():
    a = tf.placeholder(dtype=tf.float32)
    mask = a> 1
    mask.set_shape([None, None])

    b = tf.placeholder(dtype=tf.float32)
    c = tf.boolean_mask(b, mask)
    with tf.Session() as sess:
        a_val = np.ones((3,))
        a_val[2] = 3;
        b_val = a_val
        c_val = sess.run(c, feed_dict={a:a_val, b:b_val})
        print c_val

def iden_s():
    a  = tf.Variable(1, name='a')
    b = tf.Variable(2, name='b')
    with tf.name_scope('ss'):
        c = a + b
        c = tf.identity(c,'c')

    print c.name

def two_tensor_max():
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)
    c = tf.maximum(a,b )
    with tf.Session() as sess:
        a_val = np.ones((3,3))

        a_val[0,2] = 100
        b_val = np.ones((3,3))
        print a_val
        print b_val
        c_val = sess.run(c, feed_dict={a:a_val, b:b_val})
        print c_val

def py_func_s():

    def add(a, b):
        return a* b

    def tf_add(a, b):
        return tf.py_func(add, [a, b], tf.float32)

    ta = tf.placeholder(dtype=tf.float32)
    tb = tf.placeholder(dtype=tf.float32)
    tc = tf_add(ta, tb)
    with tf.Session() as sess:
        a_val = np.ones((2,2)) * 2
        b_val = a_val * 2

        print sess.run([tc], feed_dict={
            ta:a_val, tb:b_val
        })

def main():
    py_func_s()

if __name__ == '__main__':
    main()