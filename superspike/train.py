import time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import GPUtil
# # Get a list of available GPUs
# gpus = GPUtil.getGPUs()
# # Select the GPU with the lowest utilization
# chosen_gpu = None
# for gpu in gpus:
#     if not chosen_gpu:
#         chosen_gpu = gpu
#     elif gpu.memoryUtil < chosen_gpu.memoryUtil:
#         chosen_gpu = gpu

# # Set CUDA device to the selected GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu.id)


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
import csv
import argparse
import numpy as np
import yaml


from load_models import load_model

import sys
sys.path.append('..')
from utils.loss import LossFn
from utils.display import SpikeCountTimeBar
from utils.config import config_train, load_sim_params, create_save_name
from data_io.load_datasets import load_dataset



class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'alpha'):
            module.alpha.data.clamp_(2/np.e, 0.995)
        if hasattr(module, 'beta'):
            module.beta.data.clamp_(2/np.e, 0.995)
clipper = ZeroOneClipper()

def train(args, model, device, train_loader, optimizer, loss_fn, sim_params, epoch, writer):
    model.test = False

    torch.cuda.empty_cache()
    with torch.autograd.set_detect_anomaly(True):
        model.train()

        correct = 0
        all_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # optimizer.zero_grad()
            data  = data.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)
            
            
            outputs = model(data) #output_times, firing_rate, avg_spk_cnt
            loss = loss_fn.forward(outputs, target)
            pred, num_correct = loss_fn.pred_label(target) #, loss_mode='first_time')
            
            correct += num_correct
            all_correct += num_correct
            
            loss.backward(retain_graph=False)
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()
            model.apply(clipper)    
            
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('running loss',
                                loss,
                                epoch * len(train_loader) + batch_idx * sim_params['learning']['batch_size'])
                acc = 100. * correct / (args.log_interval * sim_params['learning']['batch_size'])
                if batch_idx == 0:
                    acc = 100. * correct / sim_params['learning']['batch_size']
                writer.add_scalar('running acc',
                                acc,
                                epoch * len(train_loader) + batch_idx * sim_params['learning']['batch_size'])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),
                    acc))
                correct = 0.

        
        print('\n Training Accuracy: {:.2f}%'.format(100. * all_correct / len(train_loader.dataset))) #Avg output spikes: {:.4f}\n
        
        writer.add_scalar('training acc',
                        100. * all_correct / len(train_loader.dataset),
                        epoch)


def test(args, model, device, test_loader, sim_params, epoch, name, writer):
    # test always
    loss_params = sim_params['loss'].copy()
    
    kwargs = {}
    kwargs['dt'] = sim_params['simulation']['input_dt']
    
    if hasattr(model, 'num_neurons'):
        spike_count_time_bar = SpikeCountTimeBar(num_neurons=model.num_neurons, num_classes=sim_params['dataset']['num_classes'], T=sim_params['simulation']['input_T'], device=device, **kwargs)

    loss_fn = LossFn(loss_params=sim_params['loss'], num_classes=sim_params['dataset']['num_classes'], step=sim_params['simulation']['input_T'], mode='test')
    model.eval()
    test_loss = 0
    correct = 0
    avg_pred_firing_rate = 0.
    avg_other_firing_rate = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data  = data.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)
            
            # output_times, firing_rate, list_of_output_spikes
            outputs = model(data) #output_spikes
            firing_rate = outputs[1]
            test_loss += loss_fn.forward(outputs, target).sum().item()*firing_rate.shape[0]  # sum up batch loss
            
            
            if hasattr(model, 'num_neurons'):
                spike_count_time_bar.update_by_sequence_end(model.spike_counts)
         
            pred, num_correct = loss_fn.pred_label(target)

            correct += num_correct
            avg_pred_firing_rate += torch.gather(firing_rate, index=pred, dim=1).sum()
            avg_other_firing_rate += (firing_rate.sum(-1) - torch.gather(firing_rate, index=pred, dim=1).squeeze()).sum(-1)/ (firing_rate.shape[1]-1)

    test_loss /= len(test_loader.dataset)
    avg_pred_firing_rate /= len(test_loader.dataset)
    avg_other_firing_rate /= len(test_loader.dataset)
    
    if hasattr(model, 'num_neurons'):
        spike_count_time_bar.show_and_save()
        spk_cnt =  spike_count_time_bar.avg_spike_count
        print('spike_counts:', spike_count_time_bar.avg_spike_count)
    else:
        spk_cnt = None

    writer.add_scalar('Avg {} loss'.format(name),
                    test_loss,
                    epoch)
    writer.add_scalar('{} acc'.format(name),
                    100. * correct / len(test_loader.dataset),
                    epoch)
    writer.add_scalar('{} pred firing rate'.format(name),
                    avg_pred_firing_rate,
                    epoch)
    writer.add_scalar('{} other firing rate'.format(name),
                    avg_other_firing_rate,
                    epoch)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Pred firing rate: {:.4}, Other firing rate: {:.4}\n'.format( #, Avg output spikes: {:.4f}
        name, test_loss, correct, len(test_loader.dataset),
        accuracy, avg_pred_firing_rate, avg_other_firing_rate)) #num_spikes

    return accuracy, spk_cnt, avg_pred_firing_rate.data.cpu().numpy(), avg_other_firing_rate.data.cpu().numpy()


def train_main(args, kwargs):
    sim_params = load_sim_params(args)
    
    save_name = create_save_name(sim_params, args.save_name, args.add_name)
    writer = SummaryWriter(os.path.join('./summary/',sim_params['dataset']['name'], '{}'.format(save_name)))
    
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    path_to_save = os.path.join('./results/', sim_params['dataset']['name'], save_name)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    output_txt_file = os.path.join(path_to_save, 'spk_cnt.csv')
    
    save_yaml = os.path.join(path_to_save, 'config.yaml')
    with open(save_yaml, 'w') as file:
        yaml.dump(sim_params, file)
    
    
    model, load_epoch, load_acc = load_model(args, sim_params, device, test=False)
    optimized_params = model.parameters()
   
    print(model)
    for name, param in model.named_parameters():
        print(name, param.mean(), param.var(),param.max(), param.min())


    
    optimizer = optim.Adam(params=optimized_params, lr=sim_params['learning']['lr'], amsgrad = True, weight_decay=sim_params['learning']['weight_decay'])
    if args.lr_decay:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=0, last_epoch=-1)
    
    train_loader = load_dataset(args, kwargs, sim_params, 'train')
    test_loader = load_dataset(args, kwargs, sim_params, 'test')
    loss_fn = LossFn(loss_params=sim_params['loss'], num_classes=sim_params['dataset']['num_classes'], step=sim_params['simulation']['input_T'], mode='train')

    # train
    best_model = copy.deepcopy(model)
    best_acc = load_acc
    best_epoch = load_epoch
    
   
    for epoch in range(load_epoch+1, load_epoch+sim_params['learning']['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, sim_params, epoch, writer)
        acc, spk_cnt, avg_pred_firing_rate, avg_other_firing_rate = test(args, model, device, test_loader, sim_params, epoch, 'test', writer)
        
        with open(output_txt_file, 'a+', newline='') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow([epoch, acc, spk_cnt, avg_pred_firing_rate, avg_other_firing_rate])

        if args.lr_decay:
            scheduler.step()
            print('lr: {}'.format(scheduler.get_last_lr()))
            
        # if (args.save_model):
        path_to_model = './models/'
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
        
        
        if acc >= best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc
            best_epoch = epoch
            print('best_acc', best_acc)

        if args.distributed:
            torch.save ({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'acc': acc,
            'best_acc': best_acc, 
            'sim_params': sim_params,
            'model_state_dict': model.module.state_dict(),
            'best_model_state_dict': best_model.module.state_dict(),
            }, os.path.join(path_to_model, "{}.pt".format(save_name)))
        else:
            torch.save ({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'acc': acc,
            'best_acc': best_acc, 
            'sim_params': sim_params,
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model.state_dict(),
            }, os.path.join(path_to_model, "{}.pt".format(save_name)))




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser = config_train()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    if args.distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('device: ', device)
    
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}
        torch.backends.cudnn.benchmark = True
    else:
        kwargs = {}
    
    print('best_model', args.best_model)
    
    train_main(args, kwargs)

