from __future__ import print_function

import time

import matplotlib.pyplot as plt
import tensorflow as tf

from ops import initialize_tf_variables
from ops import create_summary_writer
from wavenet_model import WaveNet


def run_model(tf_model, tf_sess, number_of_runs):
    avg_duration = 0
    random_clause = tf_model.get_random_clause_tensor()
    random_conjecture = tf_model.get_random_negated_conjecture()
    random_labels = tf_model.get_random_labels()
    for run_index in range(number_of_runs):
        start_time = time.time()
        out = tf_sess.run([tf_model.loss], feed_dict={tf_model.clause_tensor: random_clause,
                                                      tf_model.negated_conjecture: random_conjecture,
                                                      tf_model.labels: random_labels})
        duration = time.time() - start_time
        avg_duration = avg_duration + duration / runs
    return avg_duration


runs = 10
number_of_clauses = [64, 128, 256, 512, 1024]
embedding_sizes = [128, 256, 512, 1024]
channel_sizes = [128, 256, 512, 1024]
block_sizes = [1]

TEST_CLAUSES = False
TEST_EMBEDDING = False
TEST_CHANNELS = False
TEST_BLOCKS = False
VIS_TENSORBOARD = True
PLOT_RESULTS = False
LOG_PATH = "/Users/phlippe/Programmierung/model_logs/WaveNet/"

if TEST_CLAUSES:
    print("Test different clause numbers...")
    time_results = []
    for clause_number in number_of_clauses:
        tf.reset_default_graph()
        sess = tf.Session()
        model = WaveNet(layer_number=7, clause_number=clause_number, embedding_size=512, block_number=1,
                        channel_size=512)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(model, sess, runs))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(number_of_clauses, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_EMBEDDING:
    print("Test different embedding sizes...")
    time_results = []
    for emb_size in embedding_sizes:
        tf.reset_default_graph()
        sess = tf.Session()
        model = WaveNet(layer_number=7, clause_number=128, embedding_size=emb_size, block_number=1,
                        channel_size=512)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(model, sess, runs))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(embedding_sizes, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_CHANNELS:
    print("Test different channel sizes...")
    time_results = []
    for ch_size in channel_sizes:
        tf.reset_default_graph()
        sess = tf.Session()
        model = WaveNet(layer_number=7, clause_number=128, embedding_size=512, block_number=1,
                        channel_size=ch_size)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(model, sess, runs))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(channel_sizes, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if TEST_BLOCKS:
    print("Test different block sizes...")
    time_results = []
    for blck_size in block_sizes:
        tf.reset_default_graph()
        sess = tf.Session()
        model = WaveNet(layer_number=7, clause_number=128, embedding_size=512, block_number=blck_size,
                        channel_size=256)
        sess.run(initialize_tf_variables())
        time_results.append(run_model(model, sess, runs))
    print(time_results)
    if PLOT_RESULTS:
        plt.plot(block_sizes, time_results, color='darkblue', linewidth=3, marker='o')
        plt.show()

if VIS_TENSORBOARD:
    print("Create summary graph for Tensorboard...")
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.reset_default_graph()
    sess = tf.Session()
    model = WaveNet(layer_number=7, clause_number=128, embedding_size=512, block_number=3,
                    channel_size=256)
    sess.run(initialize_tf_variables())

    writer = create_summary_writer(logpath=LOG_PATH, sess=sess)
    sess.run([model.loss], feed_dict={model.clause_tensor: model.get_random_clause_tensor(),
                                      model.negated_conjecture: model.get_random_negated_conjecture(),
                                      model.labels: model.get_random_labels()},
             run_metadata=run_metadata,
             options=run_options)
    writer.add_run_metadata(run_metadata, 'runtime_performance')

