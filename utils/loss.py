import torch
import torch.nn as nn
import math



class LossFn(nn.Module):
    def __init__(self, loss_params, num_classes, step, mode='test'):
        super(LossFn).__init__()
        self.num_classes = num_classes
        self.step = step
        self.loss_mode = loss_params['loss_mode']
        self.test = False if mode=='train' else True

        self.alpha_fs = loss_params['FS']['alpha']
        self.alpha_fr = loss_params['FR']['alpha']

        if mode=='train':
            self.lambda_treg = loss_params['treg']['lambda']
            self.beta_treg = loss_params['treg']['beta']
        else:
            self.lambda_treg = 0
            self.beta_treg = 0
        

    def pred_label(self, target, loss_mode=None):
        ########## input one-hot target ###########
        ########## return pred label & correct ##########
        
        self.target_label =  target.argmax(dim=1, keepdim=True)
        if loss_mode is None:
            loss_mode = self.loss_mode
        
        if loss_mode in ['first_time']:
            pred = self.output_times.argmin(dim=1, keepdim=True)
            num_correct = pred.eq(self.target_label.view_as(pred)).sum().item()
            return pred, num_correct
        else: # firing rate
            pred = self.firing_rate.argmax(dim=1, keepdim=True)
            num_correct = pred.eq(self.target_label.view_as(pred)).sum().item()
            return pred, num_correct
    

    def forward(self, outputs, target,loss_mode=None):
        ''' target  is one-hot label'''
        if loss_mode is None:
            loss_mode = self.loss_mode
        if isinstance(outputs, torch.Tensor):
            self.firing_rate = outputs
        elif isinstance(outputs, list or tuple):
            if len(outputs) == 2:
                self.output_times, self.firing_rate = outputs
            else:
                assert len(outputs) == 2
        self.device = target.device
   
        if loss_mode in ['first_time']:
            if self.test:
                regularisation = 0
            else:
                labeled_time = torch.gather(self.output_times, index=target.argmax(dim=1, keepdim=True), dim=1)
                labeled_time[labeled_time <= self.step] = 0
                regularisation = self.lambda_treg*(torch.exp(self.beta_treg *labeled_time)-1).mean() #+ spk_cnt_reg
            pred = -self.output_times + self.output_times.min(-1, keepdim=True).values

            loss = (-target * torch.log(torch.exp( self.alpha_fs*pred)/torch.exp( self.alpha_fs*pred).sum(-1, keepdim=True))).sum(-1, keepdim=True).mean() + regularisation
            
        elif loss_mode in ['firing_rate']:
            loss = (-target * torch.log(torch.exp( self.alpha_fr*self.firing_rate)/torch.exp( self.alpha_fr*self.firing_rate).sum(-1, keepdim=True))).sum(-1, keepdim=True).mean()
        else:
            assert loss_mode in ['first_time', 'firing_rate'], "loss_mode should be 'first_time', 'firing_rate'!"

        return loss   
        
