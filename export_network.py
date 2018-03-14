from CNN_embedder_network import NetType
from ops import *
from Comb_LSTM_interference import CombLSTMInterference


LOG_PATH = "Export/Log/"
LOAD_FILE = "CNN_Dense/newest_training/CNN_Dense-47999"
FREEZE_DIR = "CNN_Dense/freezed/"
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
if not os.path.exists(FREEZE_DIR):
    os.makedirs(FREEZE_DIR)


def main():
    print("="*50+"\nStart export\n"+"="*50)
    print("Log path: "+LOG_PATH)
    print("Load directory: "+LOAD_FILE)

    network = CombLSTMInterference(embedding_size=512, batch_size=32, comb_features=1024,
                                   embedding_net_type=NetType.DILATED_DENSE_BLOCK, use_conversion=False)

    init_op = initialize_tf_variables()
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, LOAD_FILE)
        freeze_graph(model_folder=FREEZE_DIR, output_node_names='CombLSTMNet/NegConjFeatures,CombLSTMNet/InitialClauses/LSTM_INITIAL/FinalState/InitStateFeatures,CombLSTMNet/CombNetwork/FinalWeights',
                     file_name='embedder_network.pb', sess=sess)
        writer = create_summary_writer(logpath=LOG_PATH, sess=sess)

if __name__ == "__main__":
    main()