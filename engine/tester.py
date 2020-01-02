import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy


class Tester():
    def __init__(self, model: torch.nn.Module):
        self.device = "cpu"
        self.model = model.to(self.device)
        self.log = {
            "nll": [],
            "softmax_dist": [],
            "prediction_ok": [],
            "input_tensor": [],
            "brier_score": 0,
            "confidence": [],
            "entropy": []}

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

    def test(self, test_dataloader=None):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():
            accuracy_count = 0
            brier = []
            pred_probs = []
            ground_truth = []

            for data in tqdm(test_dataloader):
                example, target = data

                # move data to device
                example = example.to(self.device).float()
                target = target.to(self.device)

                # predict
                prediction = self.model(example)

                # softmax
                softmax_pred = torch.nn.Softmax()(prediction).numpy()
                self.log["softmax_dist"].append(softmax_pred)

                # nll
                nll = np.log(softmax_pred) * -1
                self.log["nll"].append(nll)

                # accuracy
                predicted_class = np.argmax(softmax_pred)  # predicted class
                if predicted_class == target.item():
                    accuracy_count += 1
                    self.log["prediction_ok"].append(True)
                else:
                    self.log["prediction_ok"].append(False)

                self.log["input_tensor"].append(example)

                # multiclass brier score
                onehot_true = np.zeros(softmax_pred.size)
                try:
                    onehot_true[int(target)] = 1
                    brier.append(np.sum((softmax_pred - onehot_true)**2))
                except IndexError:
                    brier.append(float('inf'))

                # cache results
                pred_probs.append(softmax_pred)
                ground_truth.append(target.item())

            # OOD metrics
            confidence = self.ood_confidence(pred_probs)
            self.log['confidence'] = confidence

            entropy = self.entropy(pred_probs)
            self.log['entropy'] = entropy

            # Brier Score
            self.log["brier_score"] = np.mean(brier)

        return self.log
