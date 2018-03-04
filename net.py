from models.unet import *
from models.loss import focal_loss
import logging
import tensorflow as tf
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Model(object):
    def __init__(self, dataprovider, loss_name='cross_entropy', **model_kw):
        self.dataprovider = dataprovider
        self.size = 256
        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 3])
        self.y = tf.placeholder(tf.int32, [None, self.size, self.size, 1])
        self.logits = unet(x=self.x,
                           batch_norm=model_kw['batch_norm'],
                           n_class=model_kw['n_class'],
                           features=model_kw['features']
                           )
        self.predict = tf.argmax(self.logits, axis=3)
        self.loss = self.get_loss(loss_name)
        self.total_acc = self.get_acc()


    def get_loss(self, loss_name):
        if loss_name == 'cross_entropy':
            print('loss_name: ' + loss_name)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.squeeze(self.y),
                    logits=self.logits
                )
            )
            return loss
        elif loss_name == 'focal_loss':
            print('loss_name: ' + loss_name)
            y = tf.one_hot(tf.squeeze(self.y),depth=5,axis=-1)
            loss = focal_loss(
                prediction_tensor=self.logits,
                target_tensor=y)
            tf.add_to_collection('losses',loss)
            loss = tf.add_n(tf.get_collection('losses'))
            return loss


    def get_acc(self):

        y_flat = tf.reshape(self.y,[-1])
        pre_flat = tf.reshape(self.predict,[-1])
        mat = tf.confusion_matrix(
            labels=y_flat,
            predictions=pre_flat,
            num_classes=5
        )
        total_acc = tf.reduce_sum(tf.diag_part(mat)) / tf.reduce_sum(mat)

        return total_acc

    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        logging.info('Done!')
        return save_path



class Train(object):
    def __init__(self, dataprovider, model, batch_size=32):
        self.dataprovider = dataprovider
        self.batch_size = batch_size
        self.model = model


    def initialize(self, sess):
        self.learning_rate = tf.Variable(0.001)
        tf.summary.scalar('loss',self.model.loss)
        tf.summary.scalar('total_acc',self.model.total_acc)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('log/train', sess.graph)
        self.test_writer = tf.summary.FileWriter('log/test')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.model.loss)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


    def train(self, iters=100000):
        logging.info("Start optimization")
        with tf.Session() as sess:
            self.initialize(sess)
            loss_tr, loss_test, acc_tr, acc_test = 0, 0, 0, 0
            for step in range(iters):
                x_tr, y_tr = self.dataprovider.next_batch_tr(self.batch_size)
                x_test, y_test = self.dataprovider.next_batch_test(self.batch_size)
                list_tr = sess.run([self.model.loss,
                                    self.model.total_acc,
                                    self.summary_op,
                                    self.optimizer],
                                   feed_dict={
                                       self.model.x: x_tr,
                                       self.model.y: y_tr
                                   })
                list_test = sess.run([self.model.loss,
                                      self.model.total_acc,
                                      self.summary_op],
                                     feed_dict={
                                         self.model.x: x_test,
                                         self.model.y: y_test
                                     })
                self.train_writer.add_summary(list_tr[2], step)
                self.test_writer.add_summary(list_test[2], step)
                loss_tr += list_tr[0]
                loss_test += list_test[0]
                acc_tr += list_tr[1]
                acc_test += list_test[1]
                if step % 100 == 0:
                    logging.info("iter {:}, loss_tr: {:.4f}, acc_tr {:.4f}, loss_test: {:.4f}, acc_test {:.4f}"
                                 .format(step,loss_tr/100,acc_tr/100,loss_test/100,acc_test/100))
                    loss_tr,acc_tr,loss_test,acc_test = 0,0,0,0