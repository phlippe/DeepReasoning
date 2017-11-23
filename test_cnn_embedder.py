from __future__ import print_function

import time

import matplotlib.pyplot as plt

from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork
from ops import *


def run_model(tf_model, tf_sess, number_of_runs, neg_conj=True):
    avg_duration = 0
    random_clause = tf_model.clause_embedder.get_random_clause()
    random_length = tf_model.clause_embedder.get_random_length()
    if neg_conj:
        feed_dict = {
            tf_model.clause_embedder.input_clause: random_clause,
            tf_model.neg_conjecture_embedder.input_clause: random_clause,
            tf_model.clause_embedder.input_length: random_length,
            tf_model.neg_conjecture_embedder.input_length: random_length
        }
    else:
        feed_dict = {
            tf_model.clause_embedder.input_clause: random_clause,
            tf_model.clause_embedder.input_length: random_length
        }

    for run_index in range(number_of_runs):
        start_time = time.time()
        out = tf_sess.run([tf_model.weight], feed_dict=feed_dict)
        duration = time.time() - start_time
        avg_duration = avg_duration + duration / number_of_runs
    return avg_duration


LOG_PATH = "/Users/phlippe/Programmierung/model_logs/CNNEmbedder/"
# LOG_PATH = "/home/it15133/model_logs/CNNEmbedder/"
FREEZE_GRAPH = True
RUN_METADATA = False
TEST_BATCH_SIZES = False
TEST_CHANNEL_SIZES = False
TEST_CHAR_NUMBERS = False
TEST_ONLY_CLAUSE = False
TEST_TENSOR_HEIGHT = False
PLOT_RESULTS = False
BATCH_STEPS = 4
CHANNEL_SIZES = [128, 256, 512, 1024]
CHAR_NUMBERS = [20, 30, 40, 50, 75, 100]
TENSOR_HEIGHT_STEPS = BATCH_STEPS
RUNS = 5

if FREEZE_GRAPH or RUN_METADATA:
    tensor_height = 1
    batch_size = 64
    if FREEZE_GRAPH:
        print("Freeze clause graph...")
        clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", batch_size=batch_size,
                                      tensor_height=tensor_height)
        neg_conjecture_embedder = tf.placeholder(shape=[1, 1, 1, 1024], dtype="float",
                                                 name="NegConjectureInput")
        combined_network = CombNetwork(clause_embedder,
                                       tf.tile(neg_conjecture_embedder, multiples=[batch_size, 1, 1, 1]),
                                       use_neg_conj=False)
        with tf.Session() as sess:
            sess.run(initialize_tf_variables())
            saver = tf.train.Saver(max_to_keep=4)
            print("Save model...")
            save_model(saver, sess, checkpoint_dir='CNN_Clause_Embedder', model_name='CNN_Clause_Embedder', step=1)
            print("Freeze graph...")
            freeze_graph(model_folder='CNN_Clause_Embedder', output_node_names='CombNet/CalcWeights',
                         file_name='clause_embedder.pb')
        tf.reset_default_graph()
        print("Freeze negated conjecture graph...")
        neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder", batch_size=1,
                                              tensor_height=1)
        with tf.Session() as sess:
            sess.run(initialize_tf_variables())
            saver = tf.train.Saver(max_to_keep=4)
            print("Save model...")
            save_model(saver, sess, checkpoint_dir='CNN_Conj_Embedder', model_name='CNN_Conj_Embedder', step=1)
            print("Freeze graph...")
            freeze_graph(model_folder='CNN_Conj_Embedder',
                         output_node_names='NegConjectureEmbedder/channel_max_pool/EmbeddedVector',
                         file_name='neg_conjecture_embedder.pb')
        tf.reset_default_graph()

    if RUN_METADATA:
        print("Run metadata...")
        clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder", batch_size=64,
                                      tensor_height=tensor_height)
        neg_conjecture_embedder = tf.zeros(shape=[64, tensor_height, 1, 1024], dtype="float")
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder, use_neg_conj=False)
        with tf.Session() as sess:
            sess.run(initialize_tf_variables())
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            writer = create_summary_writer(logpath=LOG_PATH, sess=sess)
            result = sess.run([combined_network.weight],
                              feed_dict={
                                  combined_network.clause_embedder.input_clause: clause_embedder.get_random_clause(),
                                  combined_network.clause_embedder.input_length: clause_embedder.get_random_length()
                              },
                              run_metadata=run_metadata,
                              options=run_options)
            # ,combined_network.neg_conjecture_embedder.input_clause: neg_conjecture_embedder.get_random_clause()
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

if TEST_ONLY_CLAUSE:
    time_results = []
    emb_size = 1024
    print("Testing only clauses...")
    tf.reset_default_graph()
    sess = tf.Session()
    neg_conjecture_embedder = CNNEmbedder(embedding_size=emb_size, name="NegConjectureEmbedder", reuse_vocab=False,
                                          batch_size=1)
    sess.run(initialize_tf_variables())
    nce = sess.run([neg_conjecture_embedder.embedded_vector],
                   feed_dict={neg_conjecture_embedder.input_clause: neg_conjecture_embedder.get_random_clause(),
                              neg_conjecture_embedder.input_length: neg_conjecture_embedder.get_random_length()})
    sess.close()

    for batch_index in range(BATCH_STEPS):
        batch_size = 2 ** batch_index
        print("Run model with batch size " + str(batch_size))
        tf.reset_default_graph()
        sess = tf.Session()

        clause_embedder = CNNEmbedder(embedding_size=emb_size, name="ClauseEmbedder", batch_size=batch_size)
        combined_network = CombNetwork(clause_embedder, tf.zeros(dtype="float", shape=[batch_size, 1, 1, emb_size]),
                                       use_neg_conj=False)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(combined_network, sess, RUNS, neg_conj=False))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot([2 ** i for i in range(BATCH_STEPS)], time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_TENSOR_HEIGHT:
    time_results = []
    emb_size = 1024
    batch_size = 4
    print("Testing different tensor heights...")
    tf.reset_default_graph()

    for height_index in range(TENSOR_HEIGHT_STEPS):
        tensor_height = 2 ** height_index
        print("Run model with tensor height " + str(tensor_height) + ", batch size " +
              str(batch_size) + " and embedding size " + str(emb_size))
        tf.reset_default_graph()
        sess = tf.Session()

        clause_embedder = CNNEmbedder(embedding_size=emb_size, name="ClauseEmbedder", batch_size=batch_size,
                                      tensor_height=tensor_height)
        combined_network = CombNetwork(clause_embedder,
                                       tf.zeros(dtype="float", shape=[batch_size, tensor_height, 1, emb_size]),
                                       use_neg_conj=False)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(combined_network, sess, RUNS, neg_conj=False))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot([2 ** i for i in range(BATCH_STEPS)], time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()
