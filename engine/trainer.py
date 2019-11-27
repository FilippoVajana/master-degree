import itertools
import torch
from tqdm import tqdm
from mlflow import log_metric
import pandas as pd
from engine.runconfig import RunConfig


class GenericTrainer():
    # TODO: target device as constructor parameter
    def __init__(self, cfg: RunConfig):
        self.device = cfg.device
        self.model = cfg.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer_args['lr'],
            weight_decay=cfg.optimizer_args['weight_decay'],
            betas=cfg.optimizer_args['betas'],
            eps=cfg.optimizer_args['eps']
        )
        self.loss_fn = torch.nn.MSELoss()

    def train(self, epochs=0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        self.model = self.model.to(self.device)
        best_model = self.model.state_dict()
        best_loss = None
        train_loss = list()
        validation_loss = list()

        for epoch in tqdm(range(epochs)):
            # train loop
            tmp_loss = torch.zeros(len(train_dataloader), device=self.device)

            self.model.train()
            for idx, batch in enumerate(train_dataloader):
                b_loss = self.__train_batch(batch)
                tmp_loss[idx] = b_loss

            # update train log
            t_loss = tmp_loss.mean().item()
            tqdm.write("Train Loss: {}".format(t_loss))
            log_metric('train loss', t_loss, step=epoch)
            train_loss.append(t_loss)


            # validation loop
            if validation_dataloader is not None:
                tmp_loss = torch.zeros(len(validation_dataloader), device=self.device) 
                self.model.eval()
                with torch.no_grad():
                    for idx, batch in enumerate(validation_dataloader):
                        b_loss = self.__validate_batch(batch)
                        tmp_loss[idx] = b_loss                    
                
                # update validation log
                v_loss = tmp_loss.mean().item()
                tqdm.write("Validation Loss: {}".format(v_loss))
                log_metric('validation loss', v_loss, step=epoch)
                validation_loss.append(v_loss)

            # save checkpoint
            if best_loss is None or t_loss < best_loss:
                best_loss = t_loss
                best_model = self.model.state_dict()

        # record train data
        t_data = list(itertools.zip_longest(train_loss, validation_loss, fillvalue=0))
        df = pd.DataFrame(data=t_data, columns=['train loss', 'validation loss'])

        return best_model, df


    def __train_batch(self, batch):
        """
        Train a batch of data.
        """                 

        examples, labels = batch

        # move data to device
        examples = examples.to(self.device)
        labels = labels.to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, labels)

        # backpropagation and gradients computation
        loss.backward()

        # update weights
        self.optimizer.step()
        
        return loss.detach()


    def __validate_batch(self, batch):
        """
        Validate a batch of data. Usually for validation batch_size=1.
        """
        
        examples, targets = batch

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)
        
        return loss.detach()


# class Trainer_v1():
#     def __init__(self, model, device):
#         self.device = device

#         # set model
#         self.model = model.to(device)

#         # set default optimizer
#         lr = 0.001
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08)
         
#         # set default loss function
#         self.loss_fn = torch.nn.L1Loss()
#         # self.loss_fn = torch.nn.MSELoss()

#         # save best model parameters
#         self.best_model = model.state_dict()

#         # define log objects
#         self.log = Logger("train_log", ["t_loss", "t_psnr", "t_ssim", "v_loss", "v_psnr", "v_ssim"])
        
    
#     def run(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
#         """
#         Starts the train-validation loop.
#         """

#         if train_dataloader == None :
#             raise Exception("Invalid train dataloader.")
    

#         #########
#         # DEBUG #
#         ######### 
#         logging.warning(f"Epochs: {epochs}")
#         logging.warning(f"Batches: {len(train_dataloader)}")
#         logging.warning(f"Batch size: {train_dataloader.batch_size}")

        
#         max_psnr = 0 # used to save best model params

#         for epoch in tqdm(range(epochs)):            
#             # train loop
#             tmp_loss = torch.zeros(len(train_dataloader), device=self.device)
#             tmp_psnr = torch.zeros(len(train_dataloader), device=self.device)
#             tmp_ssim = torch.zeros(len(train_dataloader), device=self.device)

#             self.model.train()
#             for idx, batch in enumerate(train_dataloader):
#                 b_loss, b_psnr, b_ssim = self.__train_batch(batch)
#                 tmp_loss[idx] = b_loss
#                 tmp_psnr[idx] = b_psnr
#                 tmp_ssim[idx] = b_ssim

#             # update train log
#             self.log.add("t_loss", tmp_loss.mean())  
#             self.log.add("t_psnr", tmp_psnr.mean())     
#             self.log.add("t_ssim", tmp_ssim.mean())    

#             # validation loop
#             if validation_dataloader == None : 
#                 continue

#             tmp_loss = torch.zeros(len(validation_dataloader), device=self.device)
#             tmp_psnr = torch.zeros(len(validation_dataloader), device=self.device)
#             tmp_ssim = torch.zeros(len(validation_dataloader), device=self.device)

#             self.model.eval()
#             with torch.no_grad():
#                 for idx, batch in enumerate(validation_dataloader):
#                     b_loss, b_psnr, b_ssim = self.__validate_batch(batch)
#                     tmp_loss[idx] = b_loss
#                     tmp_psnr[idx] = b_psnr
#                     tmp_ssim[idx] = b_ssim
            
#             # update validation log
#             self.log.add("v_loss", tmp_loss.mean())
#             self.log.add("v_psnr", tmp_psnr.mean())
#             self.log.add("v_ssim", tmp_ssim.mean())

#             # save checkpoint
#             if tmp_psnr.mean() > max_psnr:
#                 self.best_model = self.model.state_dict()

#         return self.log
        

#     def __train_batch(self, batch):
#         """
#         Train work for a batch.
#         """                 

#         examples, targets = batch     

#         # move data to device
#         examples = examples.to(self.device)
#         targets = targets.to(self.device)

#         # reset gradient computation
#         self.optimizer.zero_grad()

#         # forward
#         predictions = self.model(examples)

#         # compute loss
#         loss = self.loss_fn(predictions, targets)

#         # backpropagation and gradients computation
#         loss.backward()

#         # update weights
#         self.optimizer.step()

#         # compute quality
#         psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])
#         ssim_t = torch.tensor([Metrics.ssim(targets[idx], p) for idx, p in enumerate(predictions)])

#         return loss.detach(), psnr_t.mean(), ssim_t.mean()


#     def __validate_batch(self, batch):
#         """
#         Validation work for a batch.
#         """
        
#         examples, targets = batch

#         # move data to device
#         examples = examples.to(self.device)
#         targets = targets.to(self.device)

#         # forward
#         predictions = self.model(examples)

#         # compute loss
#         loss = self.loss_fn(predictions, targets)

#         # compute quality
#         psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])
#         ssim_t = torch.tensor([Metrics.ssim(targets[idx], p) for idx, p in enumerate(predictions)])
        
#         return loss.detach(), psnr_t.mean(), ssim_t.mean()
