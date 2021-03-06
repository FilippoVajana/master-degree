import logging as log
import time
from engine.dataloader import ImageDataLoader
from engine.runconfig import RunConfig
import pandas as pd
from tqdm import trange
import numpy as np
import torch
from random import randint
from copy import copy


log.basicConfig(level=log.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


class GenericTrainer():
    BINOMIAL_DIST = torch.distributions.Binomial(total_count=1, probs=1)
    MC_DROPOUT_PASS = 25

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
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.dirty_labels_prob = cfg.dirty_labels
        self.BINOMIAL_DIST = torch.distributions.Binomial(
            total_count=1, probs=torch.zeros(cfg.batch_size).fill_(self.dirty_labels_prob))
        self.MC_DROPOUT_PASS = 25

        # train metrics
        self.train_logs = {
            "t_mean_accuracy": [],
            "t_mean_brier": [],
            "t_mean_entropy": [],
            "t_mean_loss": [],
            "t_mean_nll": []
        }

        # validation metrics
        self.validation_logs = {
            "v_mean_accuracy": [],
            "v_mean_brier": [],
            "v_mean_entropy": [],
            "v_mean_loss": [],
            "v_mean_nll": []
        }

        # out-of-distribution metrics and dataloader
        self.ood_logs = {
            "ov_mean_brier": [],
            "ov_mean_entropy": [],
            "ov_mean_nll": []
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
        log.info(f"do_mcdropout: {self.model.do_mcdropout}")
        log.info(f"do_transferlearn: {self.model.do_transferlearn}")

        best_model = self.model
        best_loss = None

        #log.info(f"Train with MC dropout: {self.model.do_mcdropout}")
        log.info(f"Train dataset: {len(train_dataloader.dataset)}")
        log.info(f"Validation dataset: {len(validation_dataloader.dataset)}")

        for _ in trange(epochs):
            # TRAIN LOOP
            self.model.train(True)
            t_tmp_metrics = []
            for batch in train_dataloader:
                # result === (accuracy, brier, entropy, loss)
                result = self.__train_batch(batch)
                t_tmp_metrics.append(result)

            # update train log
            t_metrics = np.vstack(t_tmp_metrics)
            for idx, k in enumerate(self.train_logs.keys()):
                self.train_logs[k].extend(list(t_metrics[:, idx]))
            # t_metrics = np.median(t_metrics, axis=0)  # epoch values
            # log.debug(f"mean epoch entropy : {t_metrics[2]}")

            # t_metrics_dict = dict(zip(self.train_logs.keys(), t_metrics))
            # for k in self.train_logs.keys():
            #     self.train_logs[k].append(t_metrics_dict[k])

            self.model.eval()
            v_tmp_metrics = []

            # VALIDATION LOOP
            with torch.no_grad():
                for batch in validation_dataloader:
                    result = self.__validate_batch(batch)
                    v_tmp_metrics.append(result)

            # update validation log
            if len(v_tmp_metrics) > 0:
                v_metrics = np.vstack(v_tmp_metrics)
                # v_metrics = np.median(v_metrics, axis=0)  # epoch values
                # v_metrics_dict = dict(
                #     zip(self.validation_logs.keys(), v_metrics))
                # for k in self.validation_logs.keys():
                #     self.validation_logs[k].append(v_metrics_dict[k])
                for idx, k in enumerate(self.validation_logs.keys()):
                    self.validation_logs[k].extend(list(v_metrics[:, idx]))

            # OOD LOOP
            ov_tmp_metrics = []

            with torch.no_grad():
                for batch in self.ood_dataloader[0]:
                    result = self.__validate_batch(batch)
                    result = (result[1], result[2], result[4])
                    ov_tmp_metrics.append(result)

            # update ood log
            ov_metrics = np.vstack(ov_tmp_metrics)
            # ov_metrics = np.median(ov_metrics, axis=0)  # epoch values
            # ov_metrics_dict = dict(
            #     zip(self.ood_logs.keys(), ov_metrics))
            # for k in self.ood_logs.keys():
            #     self.ood_logs[k].append(ov_metrics_dict[k])
            for idx, k in enumerate(self.ood_logs.keys()):
                self.ood_logs[k].extend(list(ov_metrics[:, idx]))

            # save checkpoint
            if best_loss is None or self.validation_logs["v_mean_loss"][-1] < best_loss:
                best_loss = self.validation_logs["v_mean_loss"][-1]
                best_model = copy(self.model)
                # log.info(f"Update best model (loss: {best_loss})")

        # build dataframes
        t_data = {"batch": list(range(len(train_dataloader) * epochs))}
        t_data.update(self.train_logs)
        train_df = pd.DataFrame(data=t_data).set_index('batch')

        v_data = {"batch": list(range(len(validation_dataloader) * epochs))}
        v_data.update(self.validation_logs)
        validation_df = pd.DataFrame(data=v_data).set_index('batch')

        ood_data = {"batch": list(range(len(self.ood_dataloader[0]) * epochs))}
        ood_data.update(self.ood_logs)
        ood_df = pd.DataFrame(data=ood_data).set_index('batch')

        return best_model, train_df, validation_df, ood_df

    def labels_drop_v1(self, labels: torch.Tensor) -> torch.Tensor:
        '''Randomly change the labels for a part of the original labels.
        '''
        if self.dirty_labels_prob == 0.0:
            return labels
        else:
            # random extraction
            extr = self.BINOMIAL_DIST.sample()
            labels[extr > 0] = randint(0, 9)
        return labels

    def __train_batch(self, batch):
        """
        Train over a batch of data.
        """

        # move data to device
        t_examples, t_labels = batch
        t_examples = t_examples.to(self.device)

        # drop labels
        t_labels = self.labels_drop_v1(t_labels).to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        t_predictions = self.model(t_examples)

        # compute loss
        loss = self.loss_fn(t_predictions, t_labels)

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

        # compute nll
        nll = self.get_nll(t_predictions, t_labels)

        return (accuracy, brier, entropy, loss.item(), nll)

    def __validate_batch(self, batch):
        """
        Validate over a batch of data.
        """

        t_examples, t_labels = batch

        # move data to device
        t_examples = t_examples.to(self.device)
        t_labels = t_labels.to(self.device)

        # forward
        # MC dropout loop
        if self.model.do_mcdropout == True:
            mc_out = [self.model(t_examples)for _ in range(0, self.MC_DROPOUT_PASS + 1, 1)]
            t_stack = torch.stack(mc_out, dim=2)
            t_mc_mean = t_stack.mean(dim=2)
            t_mc_std = t_stack.std(dim=2)
            t_predictions = t_mc_mean
        else:
            t_predictions = self.model(t_examples)

        # compute loss
        if t_labels.min() >= 65:
            t_labels.add_(-65)
        loss = self.loss_fn(t_predictions, t_labels)

        # compute accuracy
        accuracy = self.get_accuracy(t_predictions, t_labels)

        # compute entropy
        entropy = self.get_entropy(t_predictions)

        # compute brier
        brier = self.get_brier_score(t_predictions, t_labels)

        # compute nll
        nll = self.get_nll(t_predictions, t_labels)

        return (accuracy, brier, entropy, loss.item(), nll)

    # METRICS

    def get_accuracy(self, predictions, labels) -> float:
        t_predicted_class = predictions.argmax(dim=1)
        res = (t_predicted_class == labels).sum().float() / \
            len(t_predicted_class)
        return res.to("cpu")

    def get_entropy(self, predictions) -> float:
        t_entropy = torch.distributions.Categorical(
            torch.nn.LogSoftmax(dim=1)(predictions.detach())).entropy()
        return t_entropy.to("cpu").median()

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

    def get_nll(self, predictions, labels):
        # softmax of prediction tensor
        t_softmax = torch.nn.LogSoftmax(dim=1)(predictions.detach())

        # negative log of softmax
        t_nll = [-1 * t_softmax[idx, l].to("cpu")
                 for idx, l in enumerate(labels)]
        # t_nll = torch.log(t_softmax) * -1
        return np.median(t_nll)
