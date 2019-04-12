import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

output_graph="./projected_mnist.pb"
output_node_names="projections/output/y_projection"

sess = tf.Session()
def load_checkpoint():
    saver = tf.train.import_meta_graph('./checkpoints/model.ckpt-1001.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    saver.restore(sess, "./checkpoints/model.ckpt-1001")

    return input_graph_def

def print_nodes():
    for n in input_graph_def.node:
        print(n, "\n")


def freeze():
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session
                input_graph_def, # input_graph_def is useful for retrieving the nodes 
                output_node_names.split(",")  )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
 
sess.close()