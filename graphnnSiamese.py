import tensorflow as tf
import numpy as np

def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode), [tf.shape(X)[0], -1, N_embed])
    cur_msg = tf.nn.relu(node_val)
    for t in range(iter_level):
        Li_t = tf.matmul(msg_mask, cur_msg)
        cur_info = tf.reshape(Li_t, [-1, N_embed])
        for Wi in Wembed:
            if Wi == Wembed[-1]:
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        tot_val_t = node_val + neigh_val_t
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t
    g_embed = tf.reduce_sum(cur_msg, 1)
    output = tf.matmul(g_embed, W_output) + b_output
    return output


class GraphNN(object):
    def __init__(self, N_x, Dtype, N_embed, depth_embed, N_o, ITER_LEVEL, lr, device='/gpu:0'):
        self.NODE_LABEL_DIM = N_x
        tf.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.truncated_normal(shape=[N_x, N_embed], stddev=0.1, dtype=Dtype))
            Wembed = [tf.Variable(tf.truncated_normal(shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype))
                      for _ in range(depth_embed)]
            W_output = tf.Variable(tf.truncated_normal(shape=[N_embed, N_o], stddev=0.1, dtype=Dtype))
            b_output = tf.Variable(tf.constant(0, shape=[N_o], dtype=Dtype))
            X1 = tf.placeholder(Dtype, [None, None, N_x])
            msg1_mask = tf.placeholder(Dtype, [None, None, None])
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)
            self.embed1 = embed1

            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)

            label = tf.placeholder(Dtype, [None, ])
            self.label = label
            cos = tf.reduce_sum(embed1 * embed2, 1) / tf.sqrt(tf.reduce_sum(
                embed1 ** 2, 1) * tf.reduce_sum(embed2 ** 2, 1) + 1e-10)

            diff = -cos
            self.diff = diff
            loss = tf.reduce_mean((diff + label) ** 2)
            self.loss = loss

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer

    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        if LOAD_PATH is not None:
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1], feed_dict={self.X1: X1, self.msg1_mask: mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1: X1,
                                                                  self.X2: X2, self.msg1_mask: mask1,
                                                                  self.msg2_mask: mask2, self.label: y})
        return cur_loss

    def calc_diff(self, X1, X2, mask1, mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1: X1,
                                                              self.X2: X2, self.msg1_mask: mask1,
                                                              self.msg2_mask: mask2})
        return diff

    def train(self, X1, X2, mask1, mask2, y):
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.X1: X1,
                                                                        self.X2: X2, self.msg1_mask: mask1,
                                                                        self.msg2_mask: mask2, self.label: y})
        return loss

    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
