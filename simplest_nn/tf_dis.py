
import tensorflow as tf
import numpy as np



FLAGS = tf.app.flags.FLAGS

# For algorithm config
tf.app.flags.DEFINE_float('learning_rate',0.00003, 'Initial learning rate')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            "Steps to validate and print loss")

# For distributed
tf.app.flags.DEFINE_string("ps_hosts","",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts","",
                           "Comma-separated list of host:port pairs")

tf.app.flags.DEFINE_string("job_name", "","One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index",0, "Index of task within the job")

learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

def loss(label, pred):
    return tf.square(label- pred)

def main():

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32,
                                     initializer=tf.random_normal_initializer())
            bias = tf.get_variable("bias", [1], tf.float32,
                                   initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight)  + bias

            loss_value = loss(label, pred)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_value, global_step=global_step)

            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()

            tf.summary.scalar('cost', loss_value)
            summary_op =  tf.summary.merge_all()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                                     logdir='./checkpoint/',
                                     init_op=init_op,
                                     summary_op=None,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=60)
            with sv.managed_session(server.target) as sess:
                step = 0
                while step < 10000:
                    train_x = np.random.randn(1)
                    train_y = 2*train_x + np.random.randn(1)* 0.33 + 10
                    _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                               feed_dict={
                                                   input:train_x, label:train_y
                                               })
                    if step % steps_to_validate == 0:
                        w, b, ls = sess.run([weight, bias, loss_value])
                        print "step : %d, weight: %f, bias: %f, loss: %f" \
                              %(step, w, b, ls)






if __name__ == '__main__':
    main()