from __future__ import print_function

from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork
from ops import *

LOG_PATH = "/Users/phlippe/Programmierung/model_logs/CNNEmbedder/"

print("Build up models...")
clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder")
neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", reuse_vocab=True)
combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder)
print("Start Session...")
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=4)
    sess.run(initialize_tf_variables())
    print("Save model...")
    save_model(saver, sess, checkpoint_dir='CNN_Embedder', model_name='CNN_Embedder', step=1)
    print("Freeze graph...")
    freeze_graph(model_folder='CNN_Embedder', output_node_names='CombNet/CalcWeights')
    writer = tf.summary.FileWriter(logdir=LOG_PATH, graph=sess.graph)
    result = sess.run([combined_network.weight],
                      feed_dict={combined_network.clause_embedder.input_clause: clause_embedder.get_random_clause(),
                                 combined_network.neg_conjecture_embedder.input_clause: neg_conjecture_embedder.get_random_clause()})
    print("Result = "+str(result))
