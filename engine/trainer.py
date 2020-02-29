import logging as log
import time
from engine.dataloader import ImageDataLoader
from engine.runconfig import RunConfig
import pandas as pd
from tqdm import trange
import numpy as np
import torch
from random import randint


log.basicConfig(level=log.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


def timer(func):
    def wrapper(*args):
        start_time = time.perf_counter()
        value = func(*args)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        log.debug(
            f"[{func.__name__!r}] Execution time: {elapsed_time*1000:.4f} ms")
        return value
    return wrapper


class GenericTrainer():
    BINOMIAL_DIST = torch.distributions.Binomial(total_count=1, probs=1)

    def __init__(self, cfg: RunConfig, device: str):
        self.device = device
        self.model: torch.nn.Module = cfg.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer_args['lr'],
            weight_decay=cfg.optimizer_args['weight_decay'],
            betas=cfg.optimizer_args['betas'],
            eps=cfg.optimizer_args['eps']
        )
        self.loss_fn = torch.nn.MSELoss()
        self.dirty_labels_prob = cfg.dirty_labels
        self.BINOMIAL_DIST = torch.distributions.Binomial(
            total_count=1, probs=torch.zeros(cfg.batch_size).fill_(self.dirty_labels_prob))

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

        # out-of-distribution metrics and dataloader
        self.ood_logs = {
            "ov_mean_brier": [],
            "ov_mean_entropy": []
        }

        self.ood_dataloader = ImageDataLoader(
            data_folder='./data/no-mnist',
            batch_size=cfg.batch_size,
            shuffle=False,
            transformation=None
        ).build(train_mode=False, max_items=cfg.max_items, validation_ratio=0)

    def train(self, epochs=0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        # sanity check
        if len(validation_dataloader) <= 0:
            log.error(
                f"Validation set smaller than batch size: {len(validation_dataloader.dataset.indices)} < {validation_dataloader.batch_size}")
            raise Exception("Validation set too small")

        self.model = self.model.to(self.device)
        best_model = self.model.state_dict()
        best_loss = None

        for _ in trange(epochs):
            # TRAIN LOOP
            self.model.train()
            t_tmp_metrics = []

            for batch in train_dataloader:
                # result === (accuracy, brier, entropy, loss)
                result = self.__train_batch(batch)
                t_tmp_metrics.append(result)

            # update train log
            t_metrics = np.vstack(t_tmp_metrics)
            t_metrics = np.median(t_metrics, axis=0)  # epoch values
            log.debug(f"mean epoch entropy : {t_metrics[2]}")

            t_metrics_dict = dict(zip(self.train_logs.keys(), t_metrics))
            for k in self.train_logs.keys():
                self.train_logs[k].append(t_metrics_dict[k])

            # VALIDATION LOOP
            self.model.eval()
            v_tmp_metrics = []

            with torch.no_grad():
                for batch in validation_dataloader:
                    # result === (accuracy, brier, entropy, loss)
                    result = self.__validate_batch(batch)
                    v_tmp_metrics.append(result)

            # update validation log
            if len(v_tmp_metrics) > 0:
                v_metrics = np.vstack(v_tmp_metrics)
                v_metrics = np.median(v_metrics, axis=0)  # epoch values
                v_metrics_dict = dict(
                    zip(self.validation_logs.keys(), v_metrics))
                for k in self.validation_logs.keys():
                    self.validation_logs[k].append(v_metrics_dict[k])

            # OOD LOOP
            ov_tmp_metrics = []

            with torch.no_grad():
                for batch in self.ood_dataloader[0]:
                    result = self.__validate_batch(batch)
                    # result === (brier, entropy)
                    result = (result[1], result[2])
                    ov_tmp_metrics.append(result)

            # update ood log
            ov_metrics = np.vstack(ov_tmp_metrics)
            ov_metrics = np.median(ov_metrics, axis=0)  # epoch values
            self.ood_logs["ov_mean_brier"].append(ov_metrics[0])
            self.ood_logs["ov_mean_entropy"].append(ov_metrics[1])

            # save checkpoint
            if best_loss is None or self.validation_logs["v_mean_loss"][-1] < best_loss:
                best_loss = self.validation_logs["v_mean_loss"][-1]
                best_model = self.model.state_dict()

        # merge train and validation logs
        data = {"epoch": range(epochs)}
        data.update(self.train_logs)
        data.update(self.validation_logs)
        data.update(self.ood_logs)

        # build dataframe from logs
        df = pd.DataFrame(data=data)

        return best_model, df

    def __drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.dirty_labels_prob == 0.0:
            return labels
        else:
            # random extraction
            extr = self.BINOMIAL_DIST.sample()
            for idx, v in enumerate(extr):
                if v == 1:
                    labels[idx] = randint(0, 9)
        return labels

    def __train_batch(self, batch):
        """
        Train over a batch of data.
        """

        # move data to device
        t_examples, t_labels = batch
        t_examples = t_examples.to(self.device)

        # drop labels
        t_labels = self.__drop_labels(t_labels).to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        t_predictions = self.model(t_examples)

        # compute loss
        t_softmax_predictions = torch.nn.LogSoftmax(dim=1)(t_predictions)
        loss = self.loss_fn(t_softmax_predictions, t_labels)

        # backpropagation and gradients computation
        loss.backward()

        # update weights
        self.optimizer.step()

        # compute accuracy
        accuracy = self.get_accuracy(t_predictions, t_labels)

        # compute entropy
        entropy = self.get_entropy(t_predictions)

        # compute brier
        brier = self.get_brier_score(t_predictions, t_labels)

        # return (accuracy, brier, entropy, loss.item())
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
        if labels.min() >= 65:
            labels.add_(-65)
        loss = self.loss_fn(predictions, labels)

        # compute accuracy
        accuracy = self.get_accuracy(predictions, labels)

        # compute entropy
        entropy = self.get_entropy(predictions)

        # compute brier
        brier = self.get_brier_score(predictions, labels)

        # return (accuracy, brier, entropy, loss.item())
        return (accuracy, brier, entropy, loss.item())

    # METRICS
    # @timer
    def get_accuracy(self, predictions, labels) -> float:
        t_predicted_class = predictions.argmax(dim=1)
        res = (t_predicted_class == labels).sum().float() / \
            len(t_predicted_class)
        return res.to("cpu")

    # @timer
    def get_entropy(self, predictions) -> float:
        t_entropy = torch.distributions.Categorical(
            torch.nn.Softmax(dim=1)(predictions.detach())).entropy()
        return t_entropy.to("cpu").median()

    # @timer
    def get_brier_score(self, predictions, labels) -> float:
        onehot_true = torch.zeros(predictions.size())
        onehot_true[torch.arange(len(predictions)), labels] = 1
        # softmax of prediction tensor
        prediction_softmax = torch.nn.functional.softmax(
            predictions.detach().cpu(), 1)
        # brier score
        diff = prediction_softmax - onehot_true
        square_diff = torch.pow(diff, 2)
        brier_score = torch.sum(square_diff, dim=1)
        return brier_score.to("cpu").median()
