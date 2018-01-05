from abc import ABCMeta, abstractmethod


class ModelTrainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_model(self, batch_size, embedding_size):
        pass

    @abstractmethod
    def run_model(self, sess, model, fetches, batch):
        pass

    @abstractmethod
    def get_train_batch(self, batch_size):
        pass

    @abstractmethod
    def get_val_batch(self, batch_size):
        pass

    @abstractmethod
    def get_test_batches(self, batch_size):
        pass

    @abstractmethod
    def process_test_batches(self, weights):
        pass

    @abstractmethod
    def process_specific_loss_information(self, all_losses):
        pass
