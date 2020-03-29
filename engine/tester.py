import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from engine.runconfig import RunConfig

import logging as log
log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


class Tester():
    MC_DROPOUT_PASS = 25

    def __init__(self, model, device="cpu", is_ood=False):
        self.device = device
        self.model = model.to(device)
        self.is_ood = is_ood
        self.MC_DROPOUT_PASS = 25

        # test metrics
        self.test_logs = {
            "t_good_pred": [],
            "t_brier": [],
            "t_entropy": [],
            "t_confidence": [],
            "t_nll": []
        }

    def get_predicted_class(self, t_predictions):
        # get index of max value
        t_predicted_cls = t_predictions.argmax(dim=1)
        return t_predicted_cls.to("cpu")

    def check_prediction(self, t_predictions, t_labels):
        t_predicted_cls = t_predictions.argmax(dim=1)
        res = (t_predicted_cls == t_labels)
        return res.to("cpu")

    def get_confidence(self, t_predictions, t_labels):
        # softmax
        t_softmax = torch.nn.Softmax(dim=1)(t_predictions)

        # HACK by Daniele Ciriello
        # t_softmax[torch.arange(len(t_softmax)), t_labels]

        # get prob of ground
        return [t_softmax[idx, l].item() for idx, l in enumerate(t_labels)]

    def get_ood_confidence(self, t_predictions):
        # softmax
        t_softmax = torch.nn.Softmax(dim=1)(t_predictions)

        # prediction confidence as max prob after softmax
        t_confidence = t_softmax.max(dim=1)[0]
        return t_confidence.to("cpu").numpy()

    def get_entropy(self, predictions):
        t_entropy = torch.distributions.Categorical(
            torch.nn.LogSoftmax(dim=1)(predictions.detach())).entropy()
        return t_entropy.to("cpu")

    def get_brier_score(self, predictions, labels):
        onehot_true = torch.zeros(predictions.size())
        onehot_true[torch.arange(len(predictions)), labels] = 1

        # softmax of prediction tensor
        prediction_softmax = torch.nn.functional.softmax(
            predictions.detach(), 1)

        # brier score
        diff = prediction_softmax - onehot_true
        square_diff = torch.pow(diff, 2)
        brier_score = torch.sum(square_diff, dim=1)

        return brier_score.to("cpu")

    def get_nll(self, t_predictions, t_labels):
        # softmax of prediction tensor
        t_softmax = torch.nn.LogSoftmax(dim=1)(t_predictions.detach())

        # negative log of softmax
        t_nll = [-1 * t_softmax[idx, l] for idx, l in enumerate(t_labels)]
        return torch.tensor(t_nll).to("cpu")

    def test(self, test_dataloader=None) -> pd.DataFrame:
        """
        Tests the model.
        """
        log.info(f"Test dataset: {len(test_dataloader.dataset)}")
        self.model.eval()
        if self.model.do_mcdropout:
            # add MC dropout metrics to dictionary
            self.test_logs["t_mc_mean"] = []
            self.test_logs["t_mc_std"] = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                t_examples, t_labels = batch

                # shift labels values for OOD data
                if self.is_ood:
                    t_labels = t_labels - 65

                # move data to device
                t_examples = t_examples.to(self.device).float()
                t_labels = t_labels.to(self.device)

                # predict tensor
                if self.model.do_mcdropout == True:
                    mc_out = [self.model(t_examples)
                              for _ in range(0, self.MC_DROPOUT_PASS + 1, 1)]
                    t_stack = torch.stack(mc_out, dim=2)
                    t_mc_mean = torch.nn.Softmax(dim=1)(t_stack).mean(dim=2)
                    t_mc_std = torch.nn.Softmax(dim=1)(t_stack).std(dim=2)

                    # log mean and std for predicted class
                    class_idx = t_mc_mean.argmax(dim=1)
                    t_mean = [t_mc_mean[idx, l].item() for idx, l in enumerate(class_idx)]
                    t_std = [t_mc_std[idx, l].item() for idx, l in enumerate(class_idx)]
                    self.test_logs["t_mc_mean"].extend(t_mean)
                    self.test_logs["t_mc_std"].extend(t_std)

                    t_predictions = t_mc_mean
                else:
                    t_predictions = self.model(t_examples)

                t_accuracy = self.check_prediction(t_predictions, t_labels)
                self.test_logs["t_good_pred"].extend(t_accuracy.numpy())

                t_brier = self.get_brier_score(t_predictions, t_labels)
                self.test_logs["t_brier"].extend(t_brier.numpy())

                t_entropy = self.get_entropy(t_predictions)
                self.test_logs["t_entropy"].extend(t_entropy.numpy())

                t_nll = self.get_nll(t_predictions, t_labels)
                self.test_logs["t_nll"].extend(t_nll.numpy())

                if self.is_ood == False:
                    t_confidence = self.get_confidence(t_predictions, t_labels)
                else:
                    t_confidence = self.get_ood_confidence(t_predictions)
                self.test_logs["t_confidence"].extend(t_confidence)

        # build dataframe from logs
        df = pd.DataFrame(data=self.test_logs)

        return df
