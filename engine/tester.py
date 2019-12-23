import torch
from tqdm import tqdm
import numpy as np


class Tester():
    def __init__(self, model: torch.nn.Module):
        self.device = "cpu"
        self.model = model.to(self.device)
        self.log = {
            "nll": [],
            "softmax_dist": [],
            "prediction_ok": [],
            "input_tensor": [],
            "brier_score": 0}

    def test(self, test_dataloader=None):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():
            accuracy_count = 0
            brier = []

            for data in tqdm(test_dataloader):
                examples, targets = data

                # move data to device
                examples = examples.to(self.device).float()
                targets = targets.to(self.device)

                # predict
                predictions = self.model(examples)

                # softmax
                softmax_pred = torch.nn.Softmax()(predictions).numpy()
                self.log["softmax_dist"].append(softmax_pred)

                # nll
                nll = np.log(softmax_pred) * -1
                self.log["nll"].append(nll)

                # accuracy
                predicted_class = np.argmax(softmax_pred)  # predicted class
                if predicted_class == targets.item():
                    accuracy_count += 1
                    self.log["prediction_ok"].append(True)
                else:
                    self.log["prediction_ok"].append(False)

                self.log["input_tensor"].append(examples)

                # multiclass brier score
                onehot_true = np.zeros(softmax_pred.size)
                try:
                    onehot_true[int(targets)] = 1
                    brier.append(np.sum((softmax_pred - onehot_true)**2))
                except IndexError:
                    brier.append(float('inf'))

            self.log["brier_score"] = np.mean(brier)

        return self.log
