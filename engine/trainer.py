import torch
import torch.nn as nn

from engine import *


class GenericTrainer():
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.optimizer = None
        self.loss_fn = None

        # run metrics
        self.log = Logger("train_log", ["t_loss", "v_loss"]) # TODO: use MLflow


    def train(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """        

        #########
        # DEBUG #
        ######### 
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batches: {len(train_dataloader)}")
        logger.info(f"Batch size: {train_dataloader.batch_size}")

        best_model = self.model.state_dict()
        best_loss = None

        for epoch in tqdm(range(epochs)):            
            # train loop
            tmp_loss = torch.zeros(len(train_dataloader), device=self.device)            

            self.model.train()
            for idx, batch in enumerate(train_dataloader):
                b_loss = self.__train_batch(batch)
                tmp_loss[idx] = b_loss                

            # update train log
            self.log.add("t_loss", tmp_loss.mean())                

            # validation loop
            if validation_dataloader == None : 
                continue

            tmp_loss = torch.zeros(len(validation_dataloader), device=self.device)            

            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(validation_dataloader):
                    b_loss = self.__validate_batch(batch)
                    tmp_loss[idx] = b_loss                    
            
            # update validation log
            self.log.add("v_loss", tmp_loss.mean())
           
            # save checkpoint
            if best_loss == None or tmp_loss.mean() < best_loss:
                best_loss = tmp_loss.mean()
                best_model = self.model.state_dict()

        return self.log, best_model


    def __train_batch(self, batch):
        """
        Train a batch of data.
        """                 

        examples, targets = batch     

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)

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


class Trainer_v1():
    def __init__(self, model, device):
        self.device = device

        # set model
        self.model = model.to(device)

        # set default optimizer
        lr = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08)
         
        # set default loss function
        self.loss_fn = torch.nn.L1Loss()
        # self.loss_fn = torch.nn.MSELoss()

        # save best model parameters
        self.best_model = model.state_dict()

        # define log objects
        self.log = Logger("train_log", ["t_loss", "t_psnr", "t_ssim", "v_loss", "v_psnr", "v_ssim"])
        
    
    def run(self, epochs = 0, train_dataloader=None, validation_dataloader=None):
        """
        Starts the train-validation loop.
        """

        if train_dataloader == None :
            raise Exception("Invalid train dataloader.")
    

        #########
        # DEBUG #
        ######### 
        logging.warning(f"Epochs: {epochs}")
        logging.warning(f"Batches: {len(train_dataloader)}")
        logging.warning(f"Batch size: {train_dataloader.batch_size}")

        
        max_psnr = 0 # used to save best model params

        for epoch in tqdm(range(epochs)):            
            # train loop
            tmp_loss = torch.zeros(len(train_dataloader), device=self.device)
            tmp_psnr = torch.zeros(len(train_dataloader), device=self.device)
            tmp_ssim = torch.zeros(len(train_dataloader), device=self.device)

            self.model.train()
            for idx, batch in enumerate(train_dataloader):
                b_loss, b_psnr, b_ssim = self.__train_batch(batch)
                tmp_loss[idx] = b_loss
                tmp_psnr[idx] = b_psnr
                tmp_ssim[idx] = b_ssim

            # update train log
            self.log.add("t_loss", tmp_loss.mean())  
            self.log.add("t_psnr", tmp_psnr.mean())     
            self.log.add("t_ssim", tmp_ssim.mean())    

            # validation loop
            if validation_dataloader == None : 
                continue

            tmp_loss = torch.zeros(len(validation_dataloader), device=self.device)
            tmp_psnr = torch.zeros(len(validation_dataloader), device=self.device)
            tmp_ssim = torch.zeros(len(validation_dataloader), device=self.device)

            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(validation_dataloader):
                    b_loss, b_psnr, b_ssim = self.__validate_batch(batch)
                    tmp_loss[idx] = b_loss
                    tmp_psnr[idx] = b_psnr
                    tmp_ssim[idx] = b_ssim
            
            # update validation log
            self.log.add("v_loss", tmp_loss.mean())
            self.log.add("v_psnr", tmp_psnr.mean())
            self.log.add("v_ssim", tmp_ssim.mean())

            # save checkpoint
            if tmp_psnr.mean() > max_psnr:
                self.best_model = self.model.state_dict()

        return self.log
        

    def __train_batch(self, batch):
        """
        Train work for a batch.
        """                 

        examples, targets = batch     

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # reset gradient computation
        self.optimizer.zero_grad()

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)

        # backpropagation and gradients computation
        loss.backward()

        # update weights
        self.optimizer.step()

        # compute quality
        psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])
        ssim_t = torch.tensor([Metrics.ssim(targets[idx], p) for idx, p in enumerate(predictions)])

        return loss.detach(), psnr_t.mean(), ssim_t.mean()


    def __validate_batch(self, batch):
        """
        Validation work for a batch.
        """
        
        examples, targets = batch

        # move data to device
        examples = examples.to(self.device)
        targets = targets.to(self.device)

        # forward
        predictions = self.model(examples)

        # compute loss
        loss = self.loss_fn(predictions, targets)

        # compute quality
        psnr_t = torch.tensor([Metrics.psnr(targets[idx], p) for idx, p in enumerate(predictions)])
        ssim_t = torch.tensor([Metrics.ssim(targets[idx], p) for idx, p in enumerate(predictions)])
        
        return loss.detach(), psnr_t.mean(), ssim_t.mean()

