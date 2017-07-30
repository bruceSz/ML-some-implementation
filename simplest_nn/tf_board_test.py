# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

def main():

    x = tf.placeholder(tf.float32, name="x")
    y = tf.placeholder(tf.float32, name="y")

    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # 线性模型
    linear_model = W * x + b

    # 损失函数，最小二乘
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    # 梯度下降
    optimiizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimiizer.minimize(loss)

    # loss 加入 summary。 除了scalar， 还有其他有待探索。
    tf.summary.scalar('squared_loss', loss)

    # training 数据
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("/tmp/tf_test", sess.graph)
    tf.global_variables_initializer().run(session = sess)

    for i in range(100):
        summary, _, curr_W, curr_b, curr_loss = sess.run([merged, train, W, b, loss],
                                                         {x:x_train, y:y_train})
        if i % 20 == 0:
            print "Iteration %d, W: %s, b:%s, loss: %s"\
            % (i, curr_W, curr_b, curr_loss)
        train_writer.add_summary(summary, i)




if __name__ == '__main__':
    main()