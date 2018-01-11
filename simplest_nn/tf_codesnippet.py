
import tensorflow as tf
import numpy as np

def reduce_m():
    A = np.array([[1,2],[3,4]])
    with tf.Session() as sess:
        print sess.run(tf.reduce_mean(A))
        print sess.run(tf.reduce_mean(A, axis=0))
        print sess.run(tf.reduce_mean(A, axis=1))

def clip():
    A = np.array([[1,1,2,4],[3,4,8,5]])
    with tf.Session() as sess:
        print sess.run(tf.clip_by_value(A, 2, 5))

def select():
    A = 3
    B = tf.convert_to_tensor([1,2,3,4])
    C = tf.convert_to_tensor([1,1,1,1])
    D = tf.convert_to_tensor([0,0,0,0])

    with tf.Session() as sess:
        print sess.run(tf.select(B>2, C, D))


def main():
    #reduce_m()
    #clip()
    select()

if __name__ == '__main__':
    main()