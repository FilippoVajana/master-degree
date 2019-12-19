import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from engine.runconfig import RunConfig


class Tester():
    def __init__(self, model: torch.nn.Module):
        self.device = "cpu"
        self.model = model.to(self.device)
        self.log = {
            "nll": [],
            "softmax_dist": [],
            "prediction_ok": [],
            "input_tensor": []}

    def test(self, test_dataloader=None):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():
            accuracy_count = 0

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

        return self.log
