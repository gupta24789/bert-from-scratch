import utils
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, model, tokenizer, config, device, save_checkpoints_after_epoch = -1, is_logging = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.is_logging = is_logging
        self.save_checkpoints_after_epoch = save_checkpoints_after_epoch
        self.writer = SummaryWriter(log_dir=config['experiment_name'])
        self.loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay=config['weight_decay'])

    def compute_loss(self, nsp_logits, mlm_logits, labels):
        """
        nsp_logits : (batch, 2)
        nsp_labels : (batch, 1)

        mlm_logits : (batch, max_len, vocab_size)
        mlm_labels : (batch, max_len)
        """
        nsp_labels = labels['nsp_label'] 
        mlm_labels = labels['mlm_label']

        ## NSP loss
        nsp_logits = nsp_logits.view(-1, nsp_logits.size(-1))
        nsp_labels = nsp_labels.view(-1)  ## (batch)
        nsp_loss = self.loss_fn(nsp_logits, nsp_labels)
        # print(f"nsp loss : {nsp_loss}")

        ## MLM Loss
        mlm_logits = mlm_logits.view(-1, mlm_logits.size(-1))   ## (batch * max_len, vocab_size)
        mlm_labels = mlm_labels.view(-1)                        ## (batch * max_len)
        mlm_loss = self.loss_fn(mlm_logits, mlm_labels)
        # print(f"mlm_loss : {mlm_loss}")

        return  nsp_loss + mlm_loss

    def get_next_label(self, nsp_logits):
        ## NSP Labels
        next_pred = torch.argmax(nsp_logits, dim = -1)
        return next_pred
    
    def compute_accuracy(self, pred, true):
        accuracy = (pred == true).sum()/len(true)
        return accuracy
    
    def _step(self, inputs, labels):
    
        loss = None
        nsp_logits, mlm_logits = self.model(inputs['bert_input'],inputs["segment_input"])
        next_label = self.get_next_label(nsp_logits)

        if labels is not None:
            loss = self.compute_loss(nsp_logits, mlm_logits, labels)
            
        return next_label, loss
    
    def iteration(self, data_loader, split = "train"):

        torch.cuda.empty_cache()

        running_loss = 0.0
        running_accuracy = 0.0

        self.optimizer.zero_grad()

        # loader = tqdm(data_loader, total = len(data_loader))
        loader = data_loader
    
        for i, (inputs, labels, raw) in enumerate(loader):

            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            labels = {k:v.to(self.device) for k,v in labels.items()}

            start_time = time.time()
            next_label, loss = self._step(inputs, labels)
            time_taken = time.time() - start_time
            token_rate = float(inputs['bert_input'].view(-1).size(0)/time_taken)

            ## Calcuate Accuracy
            accuracy = self.compute_accuracy(next_label, labels['nsp_label'])

            # loader.set_description(f"Step : {i}")
            # loader.set_postfix({"Step Loss": loss.item(), "Step Accuracy : " : accuracy.item()})

            running_loss += loss.item()
            running_accuracy += accuracy.item()

            if split=="train":
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss/len(loader)
        epoch_acc = running_accuracy/len(loader)
        return epoch_loss, epoch_acc, token_rate

    def train(self, train_dl, eval_dl):

        num_epochs = self.config['n_epochs']

        loader = tqdm(range(num_epochs), total = num_epochs, leave=True, bar_format='{l_bar}{bar:4}{r_bar}')

        for ep in loader:
            
            loader.set_description(f"Epoch : {ep+1}")
            ## Train
            self.model.train()
            train_loss, train_accuracy,token_rate = self.iteration(train_dl, split="train")
            ## Eval
            self.model.eval()
            val_loss, val_accuracy,token_rate = self.iteration(eval_dl)

            loader.set_postfix({
                "Train Loss": train_loss,
                "Train Acc": train_accuracy,
                "Val loss ": val_loss,
                "val Acc": val_accuracy,
                "Token rate": f"{token_rate: 10.0f}/s"
            })

            ## Save model
            if self.save_checkpoints_after_epoch == -1 :
                utils.save_model(self.config, self.model, self.optimizer, ep)
            elif (ep+1) % self.save_checkpoints_after_epoch==0:
                utils.save_model(self.config, self.model, self.optimizer, ep)

            ## Log To tensorboard
            if self.is_logging:
                self.writer.add_scalar('train_loss', train_loss, ep)
                self.writer.add_scalar('val_accuracy', train_accuracy, ep)
                self.writer.add_scalar('val_loss', val_loss, ep)
                self.writer.add_scalar('val_accuracy', val_accuracy, ep)
                self.writer.flush()

        ## Save model after last epoch
        utils.save_model(self.config, self.model, self.optimizer, ep+1)
                

            

    






