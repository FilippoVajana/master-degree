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
    def __init__(self, model, device="cpu", is_ood=False):
        self.device = device
        self.model = model.to(device)
        self.is_ood = is_ood

        # test metrics
        self.test_logs = {
            "t_good_pred": [],
            "t_brier": [],
            "t_entropy": [],
            "t_confidence": [],
            "t_output": []
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
        t_softmax = torch.nn.LogSoftmax(dim=1)(t_predictions)

        # HACK by Daniele Ciriello
        # t_softmax[torch.arange(len(t_softmax)), t_labels]

        # get prob of ground
        return [t_softmax[idx, l].item() for idx, l in enumerate(t_labels)]

    def get_ood_confidence(self, t_predictions):
        # softmax
        t_softmax = torch.nn.LogSoftmax(dim=1)(t_predictions)

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
        prediction_softmax = torch.nn.functional.log_softmax(
            predictions.detach().cpu(), 1)

        # brier score
        diff = prediction_softmax - onehot_true
        square_diff = torch.pow(diff, 2)
        brier_score = torch.sum(square_diff, dim=1)

        return brier_score.to("cpu")

    # def get_nll(self, t_predictions):
    #     # softmax of prediction tensor
    #     t_softmax = torch.nn.Softmax(dim=1)(t_predictions)

    #     # negative log of softmax
    #     t_nll = torch.log(t_softmax) * -1
    #     return t_nll.to("cpu")

    def test(self, test_dataloader=None) -> pd.DataFrame:
        """
        Tests the model.
        """
        self.model.eval()
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
                t_predictions = self.model(t_examples)

                t_accuracy = self.check_prediction(t_predictions, t_labels)
                self.test_logs["t_good_pred"].extend(list(t_accuracy.numpy()))

                t_brier = self.get_brier_score(t_predictions, t_labels)
                self.test_logs["t_brier"].extend(list(t_brier.numpy()))

                t_entropy = self.get_entropy(t_predictions)
                self.test_logs["t_entropy"].extend(list(t_entropy.numpy()))

                if self.is_ood == False:
                    t_confidence = self.get_confidence(t_predictions, t_labels)
                else:
                    t_confidence = self.get_ood_confidence(t_predictions)
                self.test_logs["t_confidence"].extend(list(t_confidence))

                for t in t_predictions:
                    self.test_logs["t_output"].append(t.numpy())

        # build dataframe from logs
        df = pd.DataFrame(data=self.test_logs)

        return df
