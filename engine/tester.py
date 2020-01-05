import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy


class Tester():
    def __init__(self, model: torch.nn.Module):
        self.device = "cpu"
        self.model = model.to(self.device)
        self.log = {
            "predicted_class": [],
            "true_class": [],
            "prediction_arr": [],
            "input_arr": []}

    def get_predicted_class(self, prediction_tensor):
        # softmax of prediction tensor
        prediction_softmax = torch.nn.Softmax()(prediction_tensor).numpy()

        # get index of max value
        predicted_class = np.argmax(prediction_softmax)
        return predicted_class

    def accuracy(self, pred_probs, targets):
        good_count = 0
        for p, t in (pred_probs, targets):
            if np.argmax(p) == t:
                good_count = good_count + 1
        return good_count / len(pred_probs)

    def ood_confidence(self, pred_probs):
        pred_probs = np.vstack(pred_probs)
        pred_probs = np.around(pred_probs, decimals=3)
        confidence = np.amax(pred_probs, axis=1)
        return confidence

    def entropy(self, pred_probs):
        pred_probs = np.vstack(pred_probs)
        pred_probs = np.around(pred_probs, decimals=3)

        preds_entropy = []
        for pred in pred_probs:
            preds_entropy.append(entropy(pred))

        return np.vstack(preds_entropy)

    def brier_score(self, prediction_tensor, true_class):
        # ground truth one-hot encoding
        onehot_true = np.zeros(prediction_tensor.size)
        onehot_true[true_class] = 1

        # softmax of prediction tensor
        prediction_softmax = torch.nn.Softmax()(prediction_tensor).numpy()

        # brier score
        brier_score = np.sum((prediction_softmax - onehot_true)**2)
        return brier_score

    def nll(self, prediction_tensor):
        # softmax of prediction tensor
        prediction_softmax = torch.nn.Softmax()(prediction_tensor).numpy()

        # negative log of softmax
        nll = np.log(prediction_softmax) * -1
        return nll

    def test(self, test_dataloader=None):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                example, target = data

                # move data to device
                example = example.to(self.device).float()
                target = target.to(self.device)

                # predict tensor
                prediction_tensor = self.model(example)

                # log prediction tensor
                self.log["prediction_arr"].append(prediction_tensor.numpy())

                # log input tensor
                self.log["input_arr"].append(example.numpy())

                # log predicted class
                self.log["predicted_class"].append(self.get_predicted_class(
                    prediction_tensor))

                # log true class
                self.log["true_class"].append(int(target))

        return self.log
