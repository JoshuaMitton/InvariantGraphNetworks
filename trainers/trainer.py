from trainers.base_train import BaseTrain
# import tensorflow as tf
import torch
from tqdm import tqdm
import numpy as np
from utils import doc_utils

class Trainer(BaseTrain):
    def __init__(self, model, data, config):
        super(Trainer, self).__init__(model, config, data)

        # load the model from the latest checkpoint if exist
        # self.model.load(self.sess)

    def train(self):
#         for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
        self.model.net.train()
        for cur_epoch in range(0, self.config.num_epochs, 1):
            self.model.optimizer.zero_grad()
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            train_loss.backward()
            self.model.optimizer.step()
            self.model.learning_rate_scheduler.step()
#             self.model.optimizer.param_groups[0]["lr"] = max(self.model.optimizer.param_groups[0]["lr"], 0.00001)
#             print(f'lr value : {self.model.optimizer}')
#             print(f'lr value : {self.model.optimizer.param_groups[0]["lr"]}')
#             print(f'lr value : {self.model.learning_rate_scheduler.get_last_lr()}')
#             self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                test_acc, test_loss = self.test(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, test_acc, test_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)
        return self.test(cur_epoch)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize(is_train=True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct = self.train_step()
#             print(f'loss device : {loss.device}')
#             print(f'correct device : {correct.device}')
            # update results from train_step func
            total_loss += loss
            total_correct += correct

#         # save model
#         if num_epoch % self.config.save_rate == 0:
#             self.model.save(self.sess)

        loss_per_epoch = total_loss/self.data_loader.train_size
        acc_per_epoch = total_correct/self.data_loader.train_size
        print("""
        Epoch-{}  loss:{:.4f} -- acc:{:.4f}
                """.format(num_epoch, loss_per_epoch, acc_per_epoch))

        tt.close()
        return acc_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """
#         print(f'device memory cached : {torch.cuda.memory_cached(0)}')
#         print(f'device memory allocated : {torch.cuda.memory_allocated(0)}')
        graphs, labels = self.data_loader.next_batch()
        graphs = torch.from_numpy(graphs)
        labels = torch.from_numpy(labels)
        if self.config.cuda:
            graphs = graphs.cuda()
            labels = labels.cuda()
#         print(f'graphs device : {graphs.device}')
        output = self.model(graphs)
#         output = output.cpu().data
        loss = self.model.loss(output, labels)
        correct = self.model.correct_predictions(output, labels)
#         _, loss, correct = self.sess.run([self.model.train_op, self.model.loss, self.model.correct_predictions],
#                                      feed_dict={self.model.graphs: graphs, self.model.labels: labels,
#                                                 self.model.is_training: True})
        return loss, correct


    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.val_size), total=self.data_loader.val_size,
                  desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            label = np.expand_dims(label, 0)
#             loss, correct = self.sess.run([self.model.loss, self.model.correct_predictions],
#                                       feed_dict={self.model.graphs: graph, self.model.labels: label, self.model.is_training: False})
            graph = torch.from_numpy(graph)
            label = torch.from_numpy(label)
            if self.config.cuda:
                graph = graph.cuda()
                label = label.cuda()
            output = self.model(graph)
            loss = self.model.loss(output, label)
            correct = self.model.correct_predictions(output, label)
            # update metrics returned from train_step func
            total_loss += loss
            total_correct += correct

        test_loss = total_loss/self.data_loader.val_size
        test_acc = total_correct/self.data_loader.val_size

        print("""
        Val-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, test_loss, test_acc))

        tt.close()
        return test_acc, test_loss
