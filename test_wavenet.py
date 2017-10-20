import time

import tensorflow as tf

from wavenet_model import WaveNet

runs = 10
model = WaveNet(layer_number=7, clause_number=128, embedding_size=512, block_number=1, channel_size=512)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

avg_duration = 0
for run_index in xrange(runs):
    start_time = time.time()
    out = sess.run([model.loss], feed_dict={model.clause_tensor: model.get_random_clause_tensor(),
                                        model.negated_conjecture: model.get_random_negated_conjecture(),
                                        model.labels: model.get_random_labels()})
    duration = time.time() - start_time
    avg_duration = avg_duration + duration / runs

print "Evaluation took " + str(avg_duration) + "sec."
