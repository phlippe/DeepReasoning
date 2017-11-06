from __future__ import print_function

import time

import matplotlib.pyplot as plt

from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork
from ops import *


def run_model(tf_model, tf_sess, number_of_runs):
    avg_duration = 0
    for run_index in range(number_of_runs):
        start_time = time.time()
        out = tf_sess.run([tf_model.weight], feed_dict={
            tf_model.clause_embedder.input_clause: tf_model.clause_embedder.get_random_clause(),
            tf_model.neg_conjecture_embedder.input_clause: tf_model.neg_conjecture_embedder.get_random_clause()})
        duration = time.time() - start_time
        avg_duration = avg_duration + duration / number_of_runs
    return avg_duration


LOG_PATH = "/Users/phlippe/Programmierung/model_logs/CNNEmbedder/"
FREEZE_GRAPH = True
TEST_BATCH_SIZES = False
TEST_CHANNEL_SIZES = True
TEST_CHAR_NUMBERS = True
PLOT_RESULTS = True
BATCH_STEPS = 8
CHANNEL_SIZES = [128, 256, 512, 1024]
CHAR_NUMBERS = [20, 30, 40, 50, 75, 100]
RUNS = 5

if FREEZE_GRAPH:
    print("Build up models...")
    clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", batch_size=8)
    neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", reuse_vocab=True,
                                          batch_size=8)
    combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder)
    print("Start Session...")
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        saver = tf.train.Saver(max_to_keep=4)
        sess.run(initialize_tf_variables())
        print("Save model...")
        save_model(saver, sess, checkpoint_dir='CNN_Embedder', model_name='CNN_Embedder', step=1)
        print("Freeze graph...")
        freeze_graph(model_folder='CNN_Embedder', output_node_names='CombNet/CalcWeights')
        writer = tf.summary.FileWriter(logdir=LOG_PATH, graph=sess.graph)
        result = sess.run([combined_network.weight],
                          feed_dict={combined_network.clause_embedder.input_clause: clause_embedder.get_random_clause(),
                                     combined_network.neg_conjecture_embedder.input_clause: neg_conjecture_embedder.get_random_clause()},
                          run_metadata=run_metadata,
                          options=run_options)
        writer.add_run_metadata(run_metadata, 'runtime_performance')
        print("Result = " + str(result))

if TEST_BATCH_SIZES:
    time_results = []
    print("Testing batch sizes...")
    for batch_index in range(BATCH_STEPS):
        batch_size = 2 ** batch_index
        print("Run model with batch size " + str(batch_size))
        tf.reset_default_graph()
        sess = tf.Session()
        clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", batch_size=batch_size)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", reuse_vocab=True,
                                              batch_size=batch_size)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(combined_network, sess, RUNS))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot([2 ** i for i in range(BATCH_STEPS)], time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_CHANNEL_SIZES:
    time_results = []
    print("Testing channel sizes...")
    for channel_size in CHANNEL_SIZES:
        print("Run model with channel size " + str(channel_size))
        tf.reset_default_graph()
        sess = tf.Session()
        clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", channel_size=channel_size,
                                      batch_size=8)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", reuse_vocab=True,
                                              channel_size=channel_size, batch_size=8)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(combined_network, sess, RUNS))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(CHANNEL_SIZES, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_CHAR_NUMBERS:
    time_results = []
    print("Testing char numbers...")
    for char_number in CHAR_NUMBERS:
        print("Run model with char number " + str(char_number))
        tf.reset_default_graph()
        sess = tf.Session()
        clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", char_number=char_number,
                                      batch_size=8)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", reuse_vocab=True,
                                              char_number=char_number, batch_size=8)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(combined_network, sess, RUNS))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(CHAR_NUMBERS, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()
