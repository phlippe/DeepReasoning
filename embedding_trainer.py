import argparse
import datetime
import time

import numpy as np

from CNN_embedder_network import NetType
from Comb_LSTM_trainer import CombLSTMTrainer
from TPTP_train_val_files import get_TPTP_test_files, get_TPTP_train_files, convert_to_absolute_path, \
    get_TPTP_test_small, get_TPTP_train_small, get_TPTP_clause_test_files
from model_trainer import ModelTrainer
from ops import *


class EmbeddingTrainer:
    def __init__(self, model_trainer, batch_size=1024, embedding_size=1024, iterations=100000,
                 val_steps=1000, save_steps=1000, checkpoint_dir='CNN_Embedder', model_name='CNNEmbedder',
                 summary_dir='logs', val_batch_number=20, lr=0.00001, loading_model=False, load_vocab=False,
                 test_steps=-1, lr_decay_steps=25000, lr_decay_rate=0.1):

        assert issubclass(type(model_trainer),
                          ModelTrainer), "Parameter model_trainer has to be a model_trainer.ModelTrainer"
        assert batch_size > 0 and type(batch_size) is int, "The batch size has to be greater zero and an integer"
        assert embedding_size > 0 and type(
            embedding_size) is int, "The embedding size has to be greater zero and an integer"
        assert val_steps > 0 and type(
            val_steps) is int, "The frequency of validation steps has to be greater zero and an integer"

        # self.train_loader = ClauseLoader(file_list=train_files, prob_pos=0.5)
        # self.test_loader = ClauseLoader(file_list=test_files, augment=False)
        # self.train_loader.print_statistic()
        # self.test_loader.print_statistic()

        self.model = None
        self.model_trainer = model_trainer

        self.training_iter = iterations
        self.val_steps = val_steps
        self.save_steps = save_steps
        self.batch_size = batch_size
        self.val_batch_number = val_batch_number
        self.lr = lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.loading_model = loading_model
        self.load_vocab = load_vocab

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_name = model_name
        self.summary_dir = summary_dir
        self.test_folder = None
        self.test_steps = test_steps

        self.model = self.model_trainer.create_model(batch_size, embedding_size)

    def run_training(self):
        print("Start training with batch size " + str(self.batch_size) + " for " + str(
            self.training_iter) + " iterations (validation every " + str(self.val_steps) + " steps)")

        saver = tf.train.Saver(max_to_keep=4)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=self.lr,
                                                   global_step=global_step,
                                                   decay_steps=self.lr_decay_steps,
                                                   decay_rate=self.lr_decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.model.loss,
                                                                                 global_step=global_step)

        with tf.Session() as sess:
            sess.run(initialize_tf_variables())
            if self.loading_model and load_model(saver, sess, self.checkpoint_dir):
                print(" [*] Load model - SUCCESS")
            elif self.loading_model:
                print(" [!] Load model - failed...")
            elif self.load_vocab:
                vocab_saver = tf.train.Saver({"CombLSTMNet/Vocabulary/Vocabs": self.model.clause_embedder.vocab_table,
                                              "CombLSTMNet/Vocabulary/Arities": self.model.clause_embedder.arity_table})
                if load_model(vocab_saver, sess, self.checkpoint_dir):
                    print(" [*] Load vocabulary - SUCCESS")
                else:
                    print(" [!] Load vocabulary - failed...")
            else:
                print(" [*] No model was loaded...")

            self.create_summary()
            summary = tf.summary.merge_all()

            timestamp = str(datetime.datetime.now()).replace("-", "_").replace(" ", "_").split('.')[0]
            train_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "train/" + timestamp), sess.graph)
            val_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "val/" + timestamp))
            self.test_folder = os.path.join(self.summary_dir, "test/" + timestamp + "/")
            if not os.path.exists(self.test_folder):
                os.makedirs(self.test_folder)

            # TRAINING LOOP
            for training_step in range(1, self.training_iter):

                if training_step % self.save_steps == 0:
                    print("Save model...")
                    save_model(saver, sess, self.checkpoint_dir, training_step, self.model_name)
                if training_step % self.val_steps == 0:
                    print("Validate model...")
                    val_summary_str = self.run_validation(sess)
                    val_writer.add_summary(val_summary_str, training_step)
                if self.test_steps > 0 and training_step % self.test_steps == 0:
                    self.run_test(sess, training_step)

                start_time = time.time()
                batch = self.model_trainer.get_train_batch(self.batch_size)
                loss, loss_zeros, loss_ones, all_losses, sum_str, _ = \
                    self.model_trainer.run_model(sess, self.model, [self.model.loss, self.model.loss_zeros,
                                                                    self.model.loss_ones, self.model.all_losses,
                                                                    summary, optimizer], batch)

                if training_step % 10 == 0:
                    train_writer.add_summary(sum_str, training_step)
                print(
                        "Iters: [%5d|%5d], time: %4.4f, clause size: %2d|%2d, loss: %.5f, loss ones:%.5f, loss zeros:%.5f" % (
                    training_step, self.training_iter, time.time() - start_time, np.max(batch[1]), np.max(batch[3]),
                    loss, loss_ones, loss_zeros))
                self.model_trainer.process_specific_loss_information(all_losses)

    def run_validation(self, sess):
        avg_loss = np.zeros(shape=3, dtype=np.float32)
        for batch_index in range(self.val_batch_number):
            batch = self.model_trainer.get_val_batch(self.batch_size)
            loss_all, loss_ones, loss_zeros = self.model_trainer.run_model(sess, self.model,
                                                                           [self.model.loss, self.model.loss_ones,
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

    def run_test(self, sess, training_step):
        print("%" * 125)
        print("TESTING MODEL...")
        all_batches = self.model_trainer.get_test_batches(self.batch_size)
        all_weights = []
        i = 0
        for batch in all_batches:
            print("Test batch " + str(i))
            i += 1
            weights = self.model_trainer.run_model(sess, self.model, [self.model.weight], batch)
            all_weights.append(weights[0])
        s = self.model_trainer.process_test_batches(all_weights)
        with open(os.path.join(self.test_folder, "test_file_" + str(training_step) + ".txt"), 'w') as f:
            f.write(s)
        print("Finished testing model")
        print("%" * 125)

    def create_summary(self):
        tf.summary.scalar('Loss', self.model.loss)
        tf.summary.scalar('Loss ones', self.model.loss_ones)
        tf.summary.scalar('Loss zeros', self.model.loss_zeros)
        tf.summary.scalar('Highest prediction', tf.reduce_max(self.model.weight))
        tf.summary.scalar('Lowest prediction', tf.reduce_min(self.model.weight))


def start_training(args):
    net_type = NetType.STANDARD
    if args.wavenet:
        net_type = NetType.WAVENET_BLOCKS
    elif args.dense:
        net_type = NetType.DILATED_DENSE_BLOCK

    modtr = CombLSTMTrainer(
        train_files=convert_to_absolute_path(args.path + "datasets/Cluster/Training/ClauseWeight_",
                                             get_TPTP_train_files() if not args.small_files else get_TPTP_train_small()),
        val_files=convert_to_absolute_path(args.path + "datasets/Cluster/Training/ClauseWeight_",
                                           get_TPTP_test_files() if not args.small_files else get_TPTP_test_small()),
        test_files=convert_to_absolute_path(args.path + "datasets/Cluster/Training/ClauseWeight_",
                                            get_TPTP_clause_test_files()),
        num_proofs=args.num_proofs,
        num_training_clauses=args.num_training,
        num_initial_clauses=args.num_init,
        num_shuffles=args.num_shuffles,
        val_batch_number=20,
        embedding_net_type=net_type,
        wavenet_blocks=args.wv_blocks,
        wavenet_layers=args.wv_layers,
        max_clause_len=args.max_clause_len,
        max_neg_conj_len=args.max_neg_conj_len
    )
    trainer = EmbeddingTrainer(model_trainer=modtr, checkpoint_dir="CNN_LSTM", model_name="CNN_LSTM",
                               val_batch_number=20, batch_size=256, val_steps=args.val_steps,
                               save_steps=args.save_steps, lr=0.00001, load_vocab=args.load_vocab,
                               loading_model=args.load_model, test_steps=args.test_steps,
                               lr_decay_rate=args.lr_decay_rate, lr_decay_steps=args.lr_decay_steps,
                               iterations=args.iterations)
    trainer.run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an embedding network')
    parser.add_argument('-p', '--path', default="/home/phillip/", help='Base path of datasets')
    parser.add_argument('-ns', '--num_shuffles', default=4, type=int,
                        help='Number of shuffles per proof for LSTM Network')
    parser.add_argument('-iter', '--iterations', default=100000, type=int,
                        help='Number of training iterations')
    parser.add_argument('-np', '--num_proofs', default=6, type=int, help='Number of proofs per batch for LSTM Network')
    parser.add_argument('-nt', '--num_training', default=32, type=int,
                        help='Number of training clauses per proof for LSTM Network')
    parser.add_argument('-ni', '--num_init', default=32, type=int,
                        help='Number of initial clauses per proof for LSTM Network')
    parser.add_argument('-lr', '--lr', default=0.00001, type=float, help='Learning rate of model')
    parser.add_argument('-vs', '--val_steps', default=200, type=int,
                        help='After how many steps the network should be validated')
    parser.add_argument('-ts', '--test_steps', default=200, type=int,
                        help='After how many steps the network should be tested. Negative numbers mean no testing at all.')
    parser.add_argument('-ss', '--save_steps', default=600, type=int,
                        help='After how many steps the network should be saved')
    parser.add_argument('-lv', '--load_vocab', action="store_true",
                        help='If previous vocabulary should be loaded or not')
    parser.add_argument('-lm', '--load_model', action="store_true", help='If previous model should be loaded or not')
    parser.add_argument('-sf', '--small_files', action="store_true",
                        help='If only a small amount of files should be used for training and testing')
    parser.add_argument('-w', '--wavenet', action="store_true", help='If embedding model should use wavenet layers')
    parser.add_argument('-db', '--dense', action="store_true",
                        help='If embedding model should use a dilated dense block')
    parser.add_argument('-wl', '--wv_layers', default=2, type=int,
                        help='If wavenet is chosen, the number of layers per block used')
    parser.add_argument('-wb', '--wv_blocks', default=1, type=int,
                        help='If wavenet is chosen, the number of blocks used')
    parser.add_argument('-cl', '--max_clause_len', default=150, type=int,
                        help='How many tokens a clause should have as maximum (otherwise will be cut)')
    parser.add_argument('-ncl', '--max_neg_conj_len', default=150, type=int,
                        help='How many tokens a negative conjecture should have as maximum (otherwise will be cut)')
    parser.add_argument('-lrds', '--lr_decay_steps', default=25000, type=int,
                        help='Number of steps after which the learning rate should be decayed by the decay rate (see --lr_decay_rate)')
    parser.add_argument('-lrdr', '--lr_decay_rate', default=0.1, type=float,
                        help='Decay rate by which the learning rate is reduced per decay steps (see --lr_decay_steps)')

    args = parser.parse_args()

    start_training(args)
