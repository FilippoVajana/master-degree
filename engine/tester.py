import torch
import time
from tqdm import tqdm
from engine.runconfig import RunConfig


class Tester():
    def __init__(self, cfg : RunConfig):
        self.device = cfg.device
        self.model = model.to(device)        

    def test(self, test_dataloader=None):
        """
        Tests the model.
        """
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                s_time = time.time()
                examples, targets = data

                # move data to device
                examples = examples.to(self.device)
                targets = targets.to(self.device)

                # predict
                predictions = self.model(examples)
                f_time = time.time()

                # compute metrics
                loss = self.loss_fn(predictions, targets)

                # update test log
                self.log.add("loss", loss)
                self.log.add("inference_time", f_time - s_time)
                
        return self.log