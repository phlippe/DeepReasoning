import numpy as np
import time
import datetime
from TPTP_train_val_files import get_TPTP_test_files, get_TPTP_train_files, convert_to_absolute_path, \
    get_TPTP_test_small, get_TPTP_train_small

from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork
from data_loader import ClauseLoader
from ops import *


class EmbeddingTrainer:
    def __init__(self, train_files, test_files, batch_size=1024, embedding_size=1024, iterations=100000,
                 val_steps=1000, save_steps=1000, checkpoint_dir='CNN_Embedder', model_name='CNNEmbedder',
                 summary_dir='logs', val_batch_number=20, lr=0.00001, use_wavenet=False):

        self.train_loader = ClauseLoader(file_list=train_files, prob_pos=0.5)
        self.test_loader = ClauseLoader(file_list=test_files, augment=False)
        self.train_loader.print_statistic()
        self.test_loader.print_statistic()

        self.model = None

        self.training_iter = iterations
        self.val_steps = val_steps
        self.save_steps = save_steps
        self.batch_size = batch_size
        self.val_batch_number = val_batch_number
        self.lr = lr

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_name = model_name
        self.summary_dir = summary_dir

        self.create_model(batch_size, embedding_size, use_wavenet)

    def create_model(self, batch_size, embedding_size, use_wavenet):
        clause_embedder = CNNEmbedder(embedding_size=embedding_size, name="ClauseEmbedder",
                                      batch_size=batch_size, char_number=None, use_wavenet=use_wavenet)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=embedding_size, name="NegConjectureEmbedder",
                                              reuse_vocab=True, batch_size=batch_size, char_number=None,
                                              use_wavenet=use_wavenet)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder, weight0=1, weight1=1)
        self.model = combined_network

    def run_training(self):
        print("Start training with batch size " + str(self.batch_size) + " for " + str(
            self.training_iter) + " iterations (validation every " + str(self.val_steps) + " steps)")

        saver = tf.train.Saver(max_to_keep=4)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.model.loss)

        with tf.Session() as sess:
            sess.run(initialize_tf_variables())
            if False and load_model(saver, sess, self.checkpoint_dir, self.model_name):
                print(" [*] Load model - SUCCESS")
            else:
                print(" [!] Load model - failed...")

            self.create_summary()
            summary = tf.summary.merge_all()

            timestamp = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_").split('.')[0]
            train_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "train/" + timestamp), sess.graph)
            val_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "val/" + timestamp))

            # TRAINING LOOP
            for training_step in range(1, self.training_iter):

                if training_step % self.save_steps == 0:
                    print("Save model...")
                    save_model(saver, sess, self.checkpoint_dir, training_step, self.model_name)
                if training_step % self.val_steps == 0:
                    print("Validate model...")
                    val_summary_str = self.run_validation(sess)
                    val_writer.add_summary(val_summary_str, training_step)

                start_time = time.time()
                batch = self.train_loader.get_batch(self.batch_size)
                loss, loss_zeros, loss_ones, sum_str, _ = self.run_model(sess, [self.model.loss, self.model.loss_zeros,
                                                                                self.model.loss_ones, summary,
                                                                                optimizer], batch)
                if training_step % 10 == 0:
                    train_writer.add_summary(sum_str, training_step)
                print(
                "Iters: [%5d|%5d], time: %4.4f, clause size: %2d|%2d, loss: %.5f, loss ones:%.5f, loss zeros:%.5f" % (
                    training_step, self.training_iter, time.time() - start_time, np.max(batch[1]), np.max(batch[3]),
                    loss, loss_ones, loss_zeros))

    def run_model(self, sess, fetches, batch):
        input_clause, input_clause_len, input_conj, input_conj_len, labels = batch
        feed_dict = {
            self.model.clause_embedder.input_clause: input_clause,
            self.model.neg_conjecture_embedder.input_clause: input_conj,
            self.model.clause_embedder.input_length: input_clause_len,
            self.model.neg_conjecture_embedder.input_length: input_conj_len,
            self.model.labels: labels
        }
        return sess.run(fetches, feed_dict=feed_dict)

    def run_validation(self, sess):
        avg_loss = np.zeros(shape=3, dtype=np.float32)
        for batch_index in range(self.val_batch_number):
            batch = self.test_loader.get_batch(self.batch_size)
            loss_all, loss_ones, loss_zeros = self.run_model(sess, [self.model.loss, self.model.loss_ones,
                                                                    self.model.loss_zeros], batch)
            avg_loss += np.array([loss_all, loss_ones, loss_zeros])
        avg_loss = avg_loss / self.val_batch_number
        print("#" * 125)
        print("VALIDATION [%d batches] - Overall loss: %.8f, loss ones: %.8f, loss zeros: %.8f" % (
            self.val_batch_number, avg_loss[0], avg_loss[1], avg_loss[2]))
        print("#" * 125)
        val_summary = tf.Summary()
        val_summary.value.add(tag="Test - Overall loss", simple_value=avg_loss[0])
        val_summary.value.add(tag="Test - Loss ones", simple_value=avg_loss[1])
        val_summary.value.add(tag="Test - Loss zeros", simple_value=avg_loss[2])
        return val_summary

    def create_summary(self):
        tf.summary.scalar('Loss', self.model.loss)
        tf.summary.scalar('Loss ones', self.model.loss_ones)
        tf.summary.scalar('Loss zeros', self.model.loss_zeros)
        tf.summary.scalar('Highest prediction', tf.reduce_max(self.model.weight))
        tf.summary.scalar('Lowest prediction', tf.reduce_min(self.model.weight))


trainer = EmbeddingTrainer(
    train_files=convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_",
                                         get_TPTP_train_files()),
    test_files=convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_", get_TPTP_test_files()),
    checkpoint_dir="CNN_Embedder_BatchNorm",
    batch_size=256, val_steps=200, save_steps=200, use_wavenet=False, lr=0.00001)
trainer.run_training()
