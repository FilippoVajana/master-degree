import itertools
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from engine.runconfig import RunConfig
from scipy.stats import entropy


class GenericTrainer():
    def __init__(self, cfg: RunConfig, device):
        self.device = device
        self.model = cfg.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer_args['lr'],
            weight_decay=cfg.optimizer_args['weight_decay'],
            betas=cfg.optimizer_args['betas'],
            eps=cfg.optimizer_args['eps']
        )
        self.loss_fn = torch.nn.MSELoss()

        # train metrics
        self.train_logs = {
            "t_mean_accuracy": [],
            "t_mean_brier": [],
            "t_mean_entropy": [],
            "t_mean_loss": []
        }

    def train(self, epochs=0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        self.model = self.model.to(self.device)
        best_model = self.model.state_dict()
        best_loss = None
        train_loss = list()
        validation_loss = list()

        for _ in tqdm(range(epochs)):
            # TRAIN LOOP
            self.model.train()
            t_tmp_metrics = []

            for idx, batch in enumerate(train_dataloader):
                # result = (accuracy, brier, entropy, loss)
                result = self.__train_batch(batch)
                t_tmp_metrics.append(result)

            # update train log
            t_metrics = np.vstack(t_tmp_metrics)
            t_metrics = np.round(t_metrics, 4)
            t_metrics = np.mean(t_metrics, axis=0)  # epoch values
            t_metrics_dict = dict(zip(self.train_logs.keys(), t_metrics))
            for k in self.train_logs.keys():
                self.train_logs[k].append(t_metrics_dict[k])
                tqdm.write(f"{k}: {self.train_logs[k]}")

            # # validation loop
            # if validation_dataloader is not None:
            #     tmp_loss = torch.zeros(
            #         len(validation_dataloader), device=self.device)
            #     self.model.eval()
            #     with torch.no_grad():
            #         for idx, batch in enumerate(validation_dataloader):
            #             b_loss = self.__validate_batch(batch)
            #             tmp_loss[idx] = b_loss

            #     # update validation log
            #     v_loss = tmp_loss.mean().item()
            #     tqdm.write("Validation Loss: {}".format(v_loss))
            #     validation_loss.append(v_loss)

            # save checkpoint
            if best_loss is None or t_loss < best_loss:
                best_loss = t_loss
                best_model = self.model.state_dict()

        # record train data
        t_data = list(itertools.zip_longest(
            train_loss, validation_loss, fillvalue=0))
        df = pd.DataFrame(data=t_data, columns=[
            'train loss', 'validation loss'
        ])

        return best_model, df

    def __train_batch(self, batch):
        """
        Train over a batch of data.
        """

        examples, labels = batch

        # move data to device
        examples = examples.to(self.device)
        labels = labels.to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, labels)

        # backpropagation and gradients computation
        loss.backward()

        # update weights
        self.optimizer.step()

        # compute accuracy
        accuracy = self.get_accuracy(predictions, labels)

        # compute entropy
        entropy = self.get_entropy(predictions)

        # compute brier
        brier = self.get_brier_score(predictions, labels)

        return (accuracy, brier, entropy, loss.item())

    def __validate_batch(self, batch):
        """
        Validate a batch of data. Usually for validation batch_size=1.
        """

        examples, targets = batch

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)

        return loss.detach()

    def get_accuracy(self, predictions, labels):
        good_count = 0
        for p, t in zip(predictions, labels):
            if np.argmax(p.detach()) == t:
                good_count = good_count + 1
        return good_count / len(predictions)

    def get_entropy(self, predictions):
        # predictions = np.vstack(predictions.detach())
        entropy_arr = []

        for pred in predictions:
            pred_softmax = torch.nn.Softmax()(pred.detach())
            pred_entropy = entropy(pred_softmax)
            entropy_arr.append(pred_entropy)

        return np.mean(entropy_arr)

    def get_brier_score(self, predictions, labels):
        score_arr = []
        for pred, lab in zip(predictions, labels):
            # ground truth one-hot encoding
            onehot_true = np.zeros(pred.shape)
            onehot_true[lab] = 1

            # softmax of prediction tensor
            pred_softmax = torch.nn.Softmax()(pred.detach()).numpy()

            # brier score
            brier_score = np.sum((pred_softmax - onehot_true)**2)
            score_arr.append(brier_score)

        return np.mean(score_arr)
