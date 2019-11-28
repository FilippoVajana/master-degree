import torch
from tqdm import tqdm
from engine.runconfig import RunConfig


class Tester():
    def __init__(self, model : torch.nn.Module):
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
                examples = examples.to(self.device)
                targets = targets.to(self.device)

                # predict
                predictions = self.model(examples)

                # NLL loss
                self.log["nll_loss"].append(torch.nn.NLLLoss()(predictions, targets))

                # Accuracy
                for idx, p in enumerate(predictions):
                    if p == targets[idx]:
                        accuracy_count += 1 / len(test_dataloader.dataset)
                        self.log["accuracy_onehot"].append(1)
                    else:
                        self.log["accuracy_onehot"].append(0)
                
        return self.log