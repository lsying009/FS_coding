import os
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
import matplotlib.pyplot as plt
import csv

from spikingjelly.activation_based import functional

import sys
sys.path.append('..')
from utils.config import config_test, load_sim_params
from utils.display import * #SpikeHistogram, SpikeCountTimeBar, AccuracywithTime, raster_plot, raster_plot_change
from utils.loss import LossFn
from spkjelly.load_models import load_model
from data_io.load_datasets import load_dataset


def test(args, model, device, test_loader, name, sim_params, load_epoch):

    loss_fn = LossFn(loss_params=sim_params['loss'],num_classes=sim_params['dataset']['num_classes'], step=sim_params['simulation']['input_T'], mode='test')
    
    
    model.eval()
    test_loss = 0
    correct_fr = 0
    correct_fs = 0.
    len_fr = 0.
    len_fs = 0.
    avg_pred_firing_rate = 0.
    avg_other_firing_rate = 0.
    avg_pred_firing_rate_fs = 0.
    avg_other_firing_rate_fs = 0.
    
    kwargs = {}
    
    kwargs['save_name'] = args.save_name
    kwargs['path_to_save'] = args.path_to_save
    # kwargs['display'] = sim_params['display']['show']
    spike_count_time_bar = SpikeCountTimeBar(num_neurons=model.num_neurons, num_classes=sim_params['dataset']['num_classes'], T=sim_params['simulation']['input_T'], device=device, **kwargs)
    accuracy_with_time_fr = AccuracywithTime(device=device, mode='firing_rate', display=False, path_to_save=args.path_to_save)
    accuracy_with_time_fs = AccuracywithTime(device=device, mode='first_time', display=False, path_to_save=args.path_to_save)

    fr_mask = None
    fs_mask = None
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            # print(data.size())
            data  = data.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)
            
            output, firing_rate, output_spikes = model(data) #output_spikes

            test_loss += loss_fn.forward([output, firing_rate], target).item() *output.shape[0] # sum up batch loss

            pred_fr, num_correct_fr = loss_fn.pred_label(target, loss_mode='firing_rate')
            pred_fs, num_correct_fs = loss_fn.pred_label(target, loss_mode='first_time')
            # print(pred_fr.shape, len(test_loader.dataset), torch.gather(firing_rate, index=pred_fr, dim=1).sum(),  (firing_rate.sum(-1) - torch.gather(firing_rate, index=pred_fr, dim=1).squeeze()).sum(-1)/ (output.shape[1]-1))
            avg_pred_firing_rate += torch.gather(firing_rate, index=pred_fr, dim=1).sum()
            avg_other_firing_rate += (firing_rate.sum(-1) - torch.gather(firing_rate, index=pred_fr, dim=1).squeeze()).sum(-1)/ (output.shape[1]-1)
            
            avg_pred_firing_rate_fs += torch.gather(firing_rate, index=pred_fs, dim=1).sum()
            avg_other_firing_rate_fs += (firing_rate.sum(-1) - torch.gather(firing_rate, index=pred_fs, dim=1).squeeze()).sum(-1)/ (output.shape[1]-1)

            correct_fr += num_correct_fr
            correct_fs += num_correct_fs

            functional.reset_net(model)
            
            if args.show:
                # print(sim_params['dataset']['num_classes'], i)
                #sim_params['dataset']['num_classes'] + 1
                if i < 5:
                    for i in range(10):
                        raster_plot_single(output_spikes[i].cpu().numpy(), loss_fn.target_label[i].squeeze().cpu().numpy(), sim_params['loss']['loss_mode'])
                        plt.savefig(os.path.join(args.path_to_save, \
                            'spikes_{:d}_gt{:d}.png'.format(i, loss_fn.target_label[i].squeeze().cpu().numpy())))
                        # i += 1
                        plt.close()

            accuracy_with_time_fr.update_by_sequence(output, output_spikes, loss_fn.target_label, fr_mask)
            accuracy_with_time_fs.update_by_sequence(output, output_spikes, loss_fn.target_label, fs_mask)
            spike_count_time_bar.update_by_sequence_end(model.spike_counts)


    len_data = len(test_loader.dataset)
    
    
    test_loss /= len_data
    avg_pred_firing_rate /= len_data
    avg_other_firing_rate /= len_data
    
    avg_pred_firing_rate_fs /= len_data
    avg_other_firing_rate_fs /= len_data
    
    acc_fr = 100. * correct_fr / len_data
    acc_fs = 100. * correct_fs / len_data

    spike_count_time_bar.show_and_save()
    accuracy_with_time_fr.show_and_save(clip_acc=50, relative_clip=True) #np.floor(acc_fs)
    accuracy_with_time_fs.show_and_save(clip_acc=50, relative_clip=True) #np.floor(acc_fs)

    td_50 = accuracy_with_time_fs.td if sim_params['loss']['loss_mode'] == 'first_time' else accuracy_with_time_fr.td
    td_90 = accuracy_with_time_fs.calc_decision_time(clip_acc=90, relative_clip=True) if sim_params['loss']['loss_mode'] == 'first_time' \
        else accuracy_with_time_fr.calc_decision_time(clip_acc=90, relative_clip=True)

    print('\n{} set: Average loss: {:.4f}, Accuracy of FR: {}/{} ({:.2f}%), Pred firing rate: {:.4}, Other firing rate: {:.4}\n'.format( #, Avg output spikes: {:.4f}
        name, test_loss, correct_fr, len_fr,
        acc_fr, avg_pred_firing_rate, avg_other_firing_rate)) #num_spikes
    
    # if fs:
    print('\n Accuracy of FS: {}/{} ({:.2f}%), Pred firing rate: {:.4}, Other firing rate: {:.4}\n'.format( #, Avg output spikes: {:.4f}
        correct_fs, len_fs,
        acc_fs, avg_pred_firing_rate_fs, avg_other_firing_rate_fs))
    
    with open(args.output_txt_file, 'a+', newline='') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(['\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Pred firing rate: {:.4}, Other firing rate: {:.4}\n'.format( #, Avg output spikes: {:.4f}
        name, test_loss, correct_fr, len_fr,
        acc_fr, avg_pred_firing_rate, avg_other_firing_rate)])
        # if fs:
        csv_writer.writerow(['\n Accuracy of FS: {}/{} ({:.2f}%) , Pred firing rate: {:.4}, Other firing rate: {:.4}\n'.format( #, Avg output spikes: {:.4f}
            correct_fs, len_fs,
            acc_fs, avg_pred_firing_rate_fs, avg_other_firing_rate_fs)])
        
        # td = accuracy_with_time_fs.td if sim_params['loss']['loss_mode'] == 'first_time' else accuracy_with_time_fr.td
        
        csv_writer.writerow(['epoch: {}'.format(load_epoch)])

        csv_writer.writerow([round(acc_fs, 2), round(acc_fr, 2), spike_count_time_bar.avg_spike_count.round(2), td_50, td_90, \
        avg_pred_firing_rate.cpu().data.numpy().round(4), avg_other_firing_rate.cpu().data.numpy().round(4)])



def test_single_delay(args, model, device, test_loader, sim_params):
    model.eval()
    # have to be batch_size = 11 classes
    num_set = sim_params['dataset']['num_classes'] + 1 if sim_params['dataset']['name'].casefold() == 'dvsgesture'.casefold() else sim_params['dataset']['num_classes']

    num_save = 10
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data  = data.float().to(device, non_blocking=True)
            target = target.float().to(device, non_blocking=True)
            
            # assert data.shape[0] == num_set, "Input batch size must be {:d}!".format(num_set)
            
            target_label = target.argmax(dim=1, keepdim=True)
        
            # collect the first appeared label (if repeat)
            unique_label, idx, counts = torch.unique(target_label, dim=0, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
            first_indicies = ind_sorted[cum_sum]
            unique_data = data[first_indicies]
            
            functional.reset_net(model)
            if n <  num_save:
                _, _, output_spikes2 = model(unique_data)
                
                plt.clf()

                raster_plot_change(output_spikes2.cpu().numpy(), unique_label.squeeze().cpu().numpy(), unique_label.squeeze().cpu().numpy(), 0) 
                plt.savefig(os.path.join(args.path_to_save, \
                    'single_{:d}.png'.format(n)))
                if args.show:
                    plt.show()
                plt.close()
                n += 1
            
 

if __name__ == '__main__':
    
    parser = config_test()
    args = parser.parse_args()
     
    torch.manual_seed(args.seed)
    

    if args.distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    sim_params = load_sim_params(args, test=True)
    test_loader = load_dataset(args, kwargs, sim_params, 'test')
    
    model, load_epoch, _ = load_model(args, sim_params, device, test=True)

    args.save_name = args.load_pt_file.split('/')[-1].rsplit('.', 1)[0]
    dataset_name = args.load_pt_file.split('/')[-1].split('_')[0]

    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    path_to_save = os.path.join('../results/', dataset_name, args.save_name)
    # print(path_to_save, not os.path.exists(path_to_save))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    args.path_to_save = path_to_save
    args.output_txt_file = os.path.join(path_to_save, 'result.csv')
    # print(args.output_txt_file)
    
    print('batch_size', args.test_batch_size)
    print('best_model', args.best_model)
    
    if args.test_single_delay:
        test_single_delay(args, model, device, test_loader, sim_params)
    else:
        print(model)
        test(args, model, device, test_loader, 'test', sim_params, load_epoch)    


    
        
