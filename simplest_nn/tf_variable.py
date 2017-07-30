
import tensorflow as tf

def vs_example():
    with tf.variable_scope("foo"):
        v1 = tf.get_variable('v1', [1])
        print v1.name
    with tf.variable_scope('foo', reuse = True):
        v2 = tf.get_variable('v1')
        print v2.name

    assert v1 is v2


def ns_example():
    with tf.variable_scope('foo') :
        with tf.name_scope('ns'):
            a = tf.get_variable('a',[1])
            b = a+ 1
            print a.name
            print b.name
            print b.op.name


def main():
    ns_example()


if __name__ == '__main__':
    main()