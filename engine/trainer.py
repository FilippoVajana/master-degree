import itertools
import torch
import numpy as np
from tqdm import tqdm, trange
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

        # validation metrics
        self.validation_logs = {
            "v_mean_accuracy": [],
            "v_mean_brier": [],
            "v_mean_entropy": [],
            "v_mean_loss": []
        }

    def train(self, epochs=0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        self.model = self.model.to(self.device)
        best_model = self.model.state_dict()
        best_loss = None

        for _ in trange(epochs):
            # TRAIN LOOP
            self.model.train()
            t_tmp_metrics = []

            for batch in train_dataloader:
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
                # tqdm.write(f"{k}: {t_metrics_dict[k]}")

            # VALIDATION LOOP
            self.model.eval()
            v_tmp_metrics = []

            with torch.no_grad():
                for batch in validation_dataloader:
                    # result = (accuracy, brier, entropy, loss)
                    result = self.__validate_batch(batch)
                    v_tmp_metrics.append(result)

            # update validation log
            v_metrics = np.vstack(v_tmp_metrics)
            v_metrics = np.round(v_metrics, 4)
            v_metrics = np.mean(v_metrics, axis=0)  # epoch values
            v_metrics_dict = dict(zip(self.validation_logs.keys(), v_metrics))
            for k in self.validation_logs.keys():
                self.validation_logs[k].append(v_metrics_dict[k])
                # tqdm.write(f"{k}: {v_metrics_dict[k]}")

            # save checkpoint
            if best_loss is None or self.validation_logs["v_mean_loss"][-1] < best_loss:
                best_loss = self.validation_logs["v_mean_loss"][-1]
                best_model = self.model.state_dict()

        # merge train and validation logs
        data = {"epoch": range(epochs)}
        data.update(self.train_logs)
        data.update(self.validation_logs)
        # self.train_logs.update(self.validation_logs)

        # build dataframe from logs
        df = pd.DataFrame(data=data)

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
        Validate over a batch of data.
        """

        examples, labels = batch

        # move data to device
        examples = examples.to(self.device)
        labels = labels.to(self.device)

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, labels)

        # compute accuracy
        accuracy = self.get_accuracy(predictions, labels)

        # compute entropy
        entropy = self.get_entropy(predictions)

        # compute brier
        brier = self.get_brier_score(predictions, labels)

        return (accuracy, brier, entropy, loss.item())

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
