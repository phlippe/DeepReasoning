import numpy as np
from ops import *
from glob import glob
import time


def test_net(file_name, batch_size, clause_length=60, number_of_runs=50):
    graph = load_frozen_graph(file_name)
    input_batch_data = graph.get_tensor_by_name("prefix/CombLSTMNet/ClauseEmbedder/InputClause:0")
    input_batch_length = graph.get_tensor_by_name("prefix/CombLSTMNet/ClauseEmbedder/InputClauseLength:0")
    input_init_features = graph.get_tensor_by_name("prefix/CombLSTMNet/CombNetwork/InitStateFeaturesPlaceholder:0")
    input_conj_features = graph.get_tensor_by_name("prefix/CombLSTMNet/CombNetwork/NegConjFeaturesPlaceholder:0")

    output_weights = graph.get_tensor_by_name("prefix/CombLSTMNet/CombNetwork/FinalWeights:0")

    feed_dict = {
        input_batch_data: np.ones(shape=[batch_size, clause_length])*5,
        input_batch_length: np.ones(shape=[batch_size])*clause_length,
        input_init_features: np.zeros(shape=[1024]),
        input_conj_features: np.zeros(shape=[1024])
    }

    with tf.Session(graph=graph) as sess:
        runtime = list()
        for _ in range(number_of_runs):
            start_time = time.time()
            _ = sess.run(output_weights, feed_dict=feed_dict)
            duration = time.time() - start_time
            runtime.append(duration)

    avg_runtime = sum(runtime) / len(runtime)
    print("Average runtime: "+str(avg_runtime))


all_network_files = sorted(glob("CNN_Dense/freezed/*.pb"))
for netfile in all_network_files:
    tf.reset_default_graph()
    batch_size = int(netfile.split("_")[-1].split(".")[0])
    print("="*50+"\nBatch size: "+str(batch_size)+"\n")
    test_net(netfile, batch_size=batch_size)