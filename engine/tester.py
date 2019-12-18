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
            "nll_loss" : [],
            "accuracy" : 0.0,
            "accuracy_onehot" : []}

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
                examples = examples.to(self.device).squeeze(dim=0).float()
                targets = targets.to(self.device)

                # predict
                predictions = self.model(examples)

                # NLL loss
                nll = torch.nn.NLLLoss()(predictions, targets)
                self.log["nll_loss"].append(nll)

                # Accuracy
                smax = torch.nn.Softmax()(predictions)
                prediction_arr = smax.numpy()
                predicted_class = np.argmax(prediction_arr)


                ### DEBUG
                # import matplotlib.pyplot as plt
                # plt.imshow(examples.squeeze().numpy())
                # plt.xlabel(f"Predicted: {predicted_class} " + f"Ground Truth: {targets.item()}")
                # plt.show()

                # plt.scatter(x=range(0,10), y=smax)
                # plt.show()
                ### DEBUG

                
                if predicted_class == targets.item():
                    accuracy_count += 1
                    self.log["accuracy_onehot"].append(1)
                else:
                    self.log["accuracy_onehot"].append(0)

                self.log["accuracy"] = accuracy_count / len(test_dataloader.dataset)
                
        return self.log