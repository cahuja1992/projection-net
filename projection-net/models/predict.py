from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import time
import os

#As our end goal is minimize the network, therefore turning off the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mnist = input_data.read_data_sets("./MNIST-data")
frozen_graph = "./projected_mnist.pb"

with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,input_map=None,return_elements=None,name="")
    y_pred = graph.get_tensor_by_name("projections/output/y_projection:0")
    x= graph.get_tensor_by_name("Placeholder:0")
    sess= tf.Session(graph=graph)

    x_batch, y_batch = mnist.test.next_batch(1000)
    start = time.time()
    result=sess.run(y_pred, feed_dict={x: x_batch})
    end = time.time()

    print("Time taken = ", (end - start) * 1000)