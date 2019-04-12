from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tf_utils import *
from model import trainer_net, projection_net

flags = tf.app.flags

def main(_):
  
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  x = tf.placeholder(tf.float32, [None, 784])
  y_= tf.placeholder(tf.int64, [None])

  

  # 1. Trainer Model Loss
  y_conv, keep_prob = trainer_net(x)
  trainer_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
  trainer_loss = tf.reduce_mean(trainer_loss)

  # 2. Projection Model Loss
  y_P = projection_net(x, T = FLAGS.projection_T, d = FLAGS.projection_D)
  projection_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_P)
  projection_loss = tf.reduce_mean(projection_loss)

  # 3. Trainer2Projection Model Loss
  entropy_loss = tf.losses.cosine_distance(y_P, y_conv,1)
  entropy_loss = tf.reduce_mean(entropy_loss)

  # 4. Combined loss 
  # loss = tf.reshape([trainer_loss, projection_loss, tf.log(entropy_loss)], [1,3])
  loss = trainer_loss + projection_loss + tf.log(entropy_loss)

  # W_loss = weight_variable([3,1])
  # y_loss = tf.matmul(loss, W_loss)
    
  # train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

  # Define training procedure
  global_step = tf.Variable(0, trainable=False)
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
  optimizer = tf.train.AdamOptimizer(0.001)
  train_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

  with tf.name_scope('trainer_accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
  
  with tf.name_scope('projection_accuracy'):
    correct_prediction2 = tf.equal(tf.argmax(y_P, 1), y_)
    correct_prediction2 = tf.cast(correct_prediction2, tf.float32)
    accuracy_P = tf.reduce_mean(correct_prediction2)

  logs = 'logs'
  print('Saving graph to: %s' % logs)
  train_writer = tf.summary.FileWriter(logs)
  train_writer.add_graph(tf.get_default_graph())


  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(FLAGS.TRAIN_ITER):
      batch = mnist.train.next_batch(FLAGS.BATCHSIZE)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
         
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      if i % 1000 == 0:
        batch_test = mnist.test.next_batch(FLAGS.BATCHSIZE)
        print('test accuracy %f' % accuracy_P.eval(feed_dict={x:batch_test[0] , y_: batch_test[1], keep_prob: 1.0}))
        
        save_path = saver.save(sess, "./checkpoints/model.ckpt", global_step=global_step)
        print("Model saved in path: %s" % save_path)

    acc = 0
    for i in range(FLAGS.TEST_ITER):
        batch_test = mnist.test.next_batch(FLAGS.BATCHSIZE)
        acc = acc + accuracy_P.eval(feed_dict={x:batch_test[0] , y_: batch_test[1], keep_prob: 1.0})
    print(acc/FLAGS.TEST_ITER)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--projection_type', type=str,
                      default='random',
                      help='Type of projection [random, lsh]') 
                     
  parser.add_argument('--projection_T', type=int,
                      default=100,
                      help='Number of projections')
  parser.add_argument('--projection_D', type=int,
                      default=8,
                      help='Dimension of every projection')
  parser.add_argument('--lr', type=float,
                      default=5e-5,
                      help='Learning rate for adam optimizer')
  parser.add_argument('--BATCHSIZE', type=int,
                      default=128,
                      help='Directory for storing input data')
  parser.add_argument('--TRAIN_ITER', type=int,
                      default=10000,
                      help='Directory for storing input data')
  parser.add_argument('--TEST_ITER', type=int,
                      default=1000,
                      help='Directory for storing input data')
  parser.add_argument('--train', type=bool,
                      default=False,
                      help='Number of projections')
  parser.add_argument('--freeze', type=bool,
                      default=False,
                      help='Number of projections')
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)