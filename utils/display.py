import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import os
import csv
 
def plot_event_tensor_2d(times, units, save_name, class_id):
    '''times [ms], units (channels)'''
    matplotlib.rc('axes', labelsize=16) 
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # print(events_pos[:10,:])#, pos_mask.shape, events_pos.shape, events_neg.shape)
    ax.scatter(times, units, color = "#301E67", marker='.',s=1) #

    ax.set_xlabel('t[ms]')
    ax.set_ylabel('x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('Class: {:d}'.format(class_id))
    
    plt.gca().set_position([0.1, 0.1, 0.85, 0.85])
    # plt.savefig("./results/events_{}.png".format(save_name), dpi=300)
    plt.show()
  



class Statistics:
    def __init__(self, display=False, path_to_save=None, save_name=None, num_neurons=None):
        self.num_neurons = num_neurons
        self.display = display
        self.save_name = save_name
        
        if path_to_save is not None:
            self.path_to_save = path_to_save
            self.output_txt_file = os.path.join(self.path_to_save, 'result.csv')
        elif save_name is not None:
            if not os.path.exists('./results/'):
                os.makedirs('./results/')
            self.path_to_save = os.path.join('./results/', self.save_name)
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
            self.output_txt_file = os.path.join(self.path_to_save, 'result.csv')
        else:
            self.path_to_save = None
                

class SpikeHistogram(Statistics):
    def __init__(self, bins, dt, display=False,
                path_to_save=None, save_name=None):
        super().__init__(display, path_to_save, save_name)
        self.fr_array = []
        self.spike_time_array = []
        self.target_first_spike_time_array = []
        self.bins = bins
        
    ## only play histogram for the last two FC layers
    def update_by_sequence(self, list_of_spike_trains, pred_ttfs, target_label):
        '''Add data from one test sequence'''
        # list_of_spike_trains = list_of_spike_trains[-2:]
        step = list_of_spike_trains[0].shape[-1]
        if not self.fr_array:
            self.step = step
            self.num_neurons = []
            # [num_sequences, n_neurons]
            for cur_spikes in list_of_spike_trains:
                n_neurons = cur_spikes.reshape(cur_spikes.shape[0], -1, step).shape[1]
                self.fr_array.append(np.empty((0,n_neurons), dtype=np.float32))
                self.spike_time_array.append(np.empty((0,n_neurons,step), dtype=np.float32))
                self.target_first_spike_time_array = np.empty((0,1), dtype=np.float32)
                self.num_neurons.append(n_neurons)

        for n, cur_spikes in enumerate(list_of_spike_trains):
            
            cur_spikes = cur_spikes.reshape(cur_spikes.shape[0], -1, step)
            cur_spike_time = torch.arange(1, step+1).to(cur_spikes.device) * cur_spikes \
                + (step+1) *(1-cur_spikes)
            cur_firing_rate = cur_spikes.mean(-1) # num_spikes per timestep per neuron
            self.fr_array[n] = np.concatenate([self.fr_array[n], cur_firing_rate.data.cpu().numpy()], axis=0)
            self.spike_time_array[n] = np.concatenate([self.spike_time_array[n], cur_spike_time.data.cpu().numpy()], axis=0)
            if n == len(list_of_spike_trains)-1:
                # B x N x T --> B x N --> B x 1
                first_spike_times, pred_indices = cur_spike_time.min(-1)
                # target_first_spike_times = first_spike_times.reshape(-1, 1)
                # B x 1
                
                target_first_spike_times = torch.gather(first_spike_times, 1, target_label) #.unsqueeze(-1).repeat(1,1,step)) 
                # correct = pred_ttfs.eq(target_label)
                # target_first_spike_times = target_first_spike_times[correct].unsqueeze(-1)
                self.target_first_spike_time_array = np.concatenate([self.target_first_spike_time_array, target_first_spike_times.data.cpu()], axis=0)
                
    def show_and_save(self):
        avg_firing_rate = []
        # sparsity = []
        all_avg_fr = 0
        for n, cur_fr_array in enumerate(self.fr_array):
            avg_cur_fr = cur_fr_array.mean()
            all_avg_fr += avg_cur_fr * self.num_neurons[n]
            avg_firing_rate.append(avg_cur_fr)
            print('Average firing rate for layer {:d}: {:.4f}'.format(n, avg_cur_fr))
        
        mean_avg_firing_rate = all_avg_fr / np.array(self.num_neurons).sum()
            
            
        for n, cur_time_array in enumerate(self.spike_time_array):
            if n == len(self.spike_time_array)-1:
                cur_time_array = cur_time_array.min(-1) 
            counts, bins = np.histogram(cur_time_array.ravel(), bins=self.bins, range=(0,self.bins))
            plt.stairs(counts/counts.sum(), bins, fill=True, alpha=0.7, label='Layer {:d}'.format(n))
        # print(counts, bins)
        plt.xlabel('Spike times (ms)')
        plt.legend()
        plt.title("Histogram of spike times") 
        plt.savefig(os.path.join(self.path_to_save,
                        'hist_spike_time.png'))
        if self.display:
            plt.show()
        plt.close()
        
        self.target_first_spike_time_array = self.target_first_spike_time_array.ravel()
        real_sum_counts = self.target_first_spike_time_array.shape[0]
        
        sorted_first_times = self.target_first_spike_time_array[self.target_first_spike_time_array<=self.step]
        len_ttfs_window = sorted_first_times.shape[0]
        sorted_first_times = np.sort(sorted_first_times) #[self.target_first_spike_time_array<=self.step]
        
        lower_bound = int(len_ttfs_window * 0.05)
        upper_bound = int(len_ttfs_window * 0.95)
        precision_dt = sorted_first_times[upper_bound] - sorted_first_times[lower_bound]
        v_post = (self.target_first_spike_time_array <= self.step).sum() / real_sum_counts
        print('Precision dt:', precision_dt)
        print('Probability of firing (v_post):', v_post)
        
        counts, bins = np.histogram(self.target_first_spike_time_array.ravel(), bins=self.bins, range=(0,self.bins))
        plt.stairs(counts/real_sum_counts, bins, fill=True, alpha=0.7)
        # print(counts, bins)
        plt.xlabel('First Spike Times (ms)')
        plt.title("Histogram of first spike times of target neuron") 
        plt.savefig(os.path.join(self.path_to_save,
                        'hist_target_ttfs.png'))
        if self.display:
            plt.show()
        plt.close()

        with open(self.output_txt_file, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            avg_firing_rate = np.round(avg_firing_rate, 4)#.squeeze()
            # sparsity = np.array(sparsity).squeeze().round(2)
            avg_firing_rate = np.append(avg_firing_rate, np.float32(mean_avg_firing_rate).round(4))
            # print(avg_firing_rate1.dtype, mean_avg_firing_rate.dtype)
            writer.writerow(['avg_firing_rate'])
            writer.writerow(avg_firing_rate)
            
            writer.writerow(['T / Precision (dt) / probability of firing (v)'])
            writer.writerow([self.step, precision_dt, v_post])
            writer.writerow(['Density of time'])
            writer.writerow(bins)
            writer.writerow(100*counts/real_sum_counts)
        f.close()
        

class SpikeCountTimeBar(Statistics):
    ''' show and save the average number of spikes for each class before decision
    / average time to make correct decisions
    '''
    def __init__(self, num_neurons, num_classes, T, device, display=False,
                path_to_save=None, save_name=None):
        super().__init__(display, path_to_save, save_name, num_neurons)
        self.num_classes = num_classes
        self.T = T
        self.device = device
        # self.loss_mode = loss_mode
        self.class_counts = torch.zeros((self.num_classes, 1),dtype=torch.int64, device=device)
        self.spike_counts = torch.zeros((self.num_classes, 1), dtype=torch.int64, device=device)
        self.spike_counts_per_layer = None
        # self.decision_times = torch.zeros((self.num_classes, 1), dtype=torch.float32, device=device)
        

    def update_by_sequence_end(self, list_of_spike_counts):
        # step = list_of_spike_trains[-1].shape[-1]
        # B = list_of_spike_trains[-1].shape[0]
        if self.spike_counts_per_layer is None:
            self.spike_counts_per_layer = torch.zeros((len(list_of_spike_counts), 1), dtype=torch.float32, device=self.device)
            self.num_sequences = 0

        for i, spike_counts in enumerate(list_of_spike_counts):
            self.spike_counts_per_layer[i] += spike_counts
        self.num_sequences += 1
     
        
    def update_by_sequence(self, list_of_spike_trains, output_times, pred, target_label):
        # add correct count to each class, 0 for incorrect prediction, 1 for correct prediction
        incorrect_mask = ~pred.eq(target_label.view_as(pred))
        src = torch.ones_like(target_label)
        src[incorrect_mask] = 0.
        self.class_counts.scatter_add_(0, target_label, src)
        
        first_times, _ = torch.min(output_times, dim=1, keepdim=True)
        
        
        # calc spike counts before the first spike time (correct prediction)
        for spike_train in list_of_spike_trains:
            spike_train_view = spike_train.view(spike_train.shape[0], -1, spike_train.shape[-1]).long()
            n_neurons = spike_train_view.shape[1]
            mask_t = torch.einsum('abc,c->abc', torch.ones_like(spike_train_view), torch.arange(self.T, device=self.device) ) 
            spike_train_view = spike_train_view.view(spike_train.shape[0], -1)
            spike_train_view[mask_t.view(spike_train.shape[0], -1) > first_times] = 0
            # print(spike_train_view, first_times)
            batch_spike_counts = spike_train_view.sum(-1, keepdim=True) # / n_neurons
            batch_spike_counts[incorrect_mask] = 0
            self.spike_counts.scatter_add_(0, target_label, batch_spike_counts)
            
            if len(self.num_neurons) < len(list_of_spike_trains):
                self.num_neurons.append(n_neurons)
                # self.num_neurons += n_neurons
            
            
        # decision time for each lass
        correct_first_times = first_times.clone().detach()
        correct_first_times[incorrect_mask] = 0.
        # print(correct_first_times, target_label)
        self.decision_times.scatter_add_(0, target_label, correct_first_times)
        # print(output_times, correct_first_times, self.decision_times)
        
    def show_and_save(self):
        self.avg_spike_count_per_layer = self.spike_counts_per_layer.cpu().data.numpy().squeeze() / self.num_sequences / self.num_neurons
        self.avg_spike_count = self.spike_counts_per_layer.cpu().data.numpy().squeeze().sum() / self.num_sequences / self.num_neurons.sum()


        # print('Average correct decision times for all classes: {:.4f} ms'.format(avg_decision_time))
        print('Average correct spike counts: {:.2f}'.format(self.avg_spike_count))

            
        if self.path_to_save is not None:
            with open(self.output_txt_file, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['N_neurons / spike count per layer'])
                writer.writerow(range(len(self.num_neurons)))
                writer.writerow(np.append(np.array(self.num_neurons), np.array(self.num_neurons).sum()))
                writer.writerow(np.append(self.avg_spike_count_per_layer.round(2), self.avg_spike_count.round(2)))
            
            f.close()
        



class AccuracywithTime(Statistics):
    def __init__(self, device, mode='firing_rate', slide_window=0, display=False,
                path_to_save=None, save_name=None):
        super().__init__(display, path_to_save, save_name)
        self.device = device
        self.mode = mode
        self.accuracy_with_time = None
        self.time_step = None
        self.correct_counts_per_step = None
        # self.correct_counts_per_step_ttfs = None
        self.num_samples = 0
        self.slide_window = slide_window if self.mode != 'first_time' else 0
        # self.decision_time_ttfs = 0
        # self.decision_time_fr = 0
    
    def spike_cnt_sliding_window(self, tensor, len, win_size): 
        iterators = [tensor[..., i:i+win_size].sum(-1) for i in range(len-win_size)]
        return torch.stack(iterators, dim=-1)

        
    def update_by_sequence(self, first_times, output_spikes, target_label, mask=None):
        # target_label = target.argmax(dim=1)
        if mask is None:
            mask = torch.ones([target_label.shape[0],1], device=target_label.device, dtype=torch.bool)
        T = output_spikes.shape[-1]
        if self.correct_counts_per_step is None:
            self.correct_counts_per_step = torch.zeros(T-self.slide_window, dtype=torch.int64, device=self.device)
            self.time_step = np.arange(0,T-self.slide_window,1)

        self.num_samples += mask.sum().item() #output_spikes.shape[0]
        
        if self.mode == 'firing_rate':
            if self.slide_window == 0:
                spk_cnt = output_spikes.cumsum(-1)
            else:
                spk_cnt = self.spike_cnt_sliding_window(output_spikes, T, self.slide_window)
            cur_pred = spk_cnt.argmax(dim=1)
            # print(spk_cnt.max(1).values, spk_cnt.max(1).values.shape)
            # print(spk_cnt.shape, cur_pred.shape,(spk_cnt.max(1).values==0).shape)
            correct = cur_pred.eq(target_label)
            correct[spk_cnt.max(1).values==0] = 0
            correct[~mask.repeat(1,correct.shape[-1])] = 0
            self.correct_counts_per_step += correct.sum(0)
            
        elif self.mode == 'first_time':
            ttfs, pred_indices =  first_times.min(dim=1)
            
            correct_mask = pred_indices.eq(target_label.view_as(pred_indices))
            # print(mask.shape, correct_mask.shape)
            ttfs = ttfs[mask.squeeze(-1) & (correct_mask) & (ttfs < T+1)].type(torch.int64)-1
            
            # print(self.correct_counts_per_step.shape[-1], max(ttfs))

            self.correct_counts_per_step.scatter_add_(0, ttfs, torch.ones_like(ttfs))


    def avg(self):
        if self.mode == 'first_time':
            self.correct_counts_per_step = self.correct_counts_per_step.cumsum_(-1)
            
        self.accuracy_with_time = (100*self.correct_counts_per_step.cpu().data.numpy().squeeze() / self.num_samples).round(2)
        th = self.accuracy_with_time.mean()
        self.avg_acc_with_time = self.accuracy_with_time[self.accuracy_with_time>=th-1e-5].mean().round(2)

    def calc_decision_time(self, clip_acc, relative_clip):
        # assert mode in ['first_time', 'firing_rate']
        # acc_with_time = self.accuracy_with_time if  mode == 'firing_rate' else self.accuracy_with_time_ttfs
        if relative_clip:
          clip_acc = np.max(self.accuracy_with_time) * clip_acc/100
        
        td = np.where(self.accuracy_with_time >= clip_acc-1e-3)
        if (self.accuracy_with_time >= clip_acc-1e-3).any():
            td = np.min(td[-1])
        else:
            td = np.inf
        return td


    def show_and_save(self, clip_acc=50, relative_clip=False, fig_name=None):
        self.avg()
        self.td = self.calc_decision_time(clip_acc, relative_clip)

        print('Time steps: ', self.time_step[-1]+1)
        print('Accuracy with time: ', self.avg_acc_with_time) #self.accuracy_with_time[-1]
        print('Time delay ({}) (Acc = {}%) with sliding window {:d}: {}'.format(self.mode, clip_acc, self.slide_window, self.td))
        
        if self.path_to_save is not None:
          plt.plot(self.time_step, self.accuracy_with_time)
          # plt.xticks(self.time_step)
          plt.xlabel('Time step')
          plt.ylabel('Accuracy')
          plt.xlim([0, self.time_step[-1]+1])
          plt.ylim([0, 101])
          plt.title("Accuracy with time") 
          
          plt.savefig(os.path.join(self.path_to_save, \
                      'accuracy_with_time_{}_{}.png'.format(self.slide_window, self.mode) if fig_name is None \
                          else '{}_{}_{}.png'.format(fig_name, self.slide_window, self.mode)))
          if self.display:
              plt.show()
          plt.close()
            

          with open(self.output_txt_file, 'a+', newline='') as f:
              writer = csv.writer(f, delimiter='\t')
              writer.writerow(['Accuracy with time'])
              # writer.writerow(self.time_step)
              writer.writerow(self.accuracy_with_time)
              writer.writerow(['Time delay ({}) (Acc = {}%) with sliding window {:d}'.format(self.mode, clip_acc, self.slide_window)])
              writer.writerow([self.td])
          f.close()
   
        
def raster_plot_stack(spike_trains, ttfs_neuron):
  """
  Generates poisson trains

  Args:
    spike_train : binary spike trains, with shape (B, N, Lt)
    n           : number of neurons

  Returns:
    Raster plot of the spike train
  """

  # find the number of all the spike trains
  B, N, T = spike_trains.shape
  time_sequence = np.arange(0, T)
  # n should smaller than N:
#   if n > N:
#     print('The number n exceeds the size of spike trains')
#     print('The number n is set to be the size of spike trains')
#     n = N
  
  # plot rater
  b = 0
#   color = iter(cm.rainbow(np.linspace(0, 1, B)))

  while b < B:
    c = 'r' if b in ttfs_neuron else 'b' #next(color)
    i = 0
    while i < N:
        if spike_trains[b, i, :].sum() > 0.:
            t_sp = T*b+time_sequence[spike_trains[b, i, :]>0.5]
            plt.plot(t_sp, i * np.ones(len(t_sp)), '.', ms=2, markeredgewidth=2, c=c, alpha=0.7)
            plt.vlines(x=[T*(b+1)], ymin=-0.5, ymax=N-0.5, colors='gray', ls='--', lw=0.5)

        i += 1
    # print(b, spike_trains[b, :, :].sum(-1))
    # plt.show()
    b += 1
  
  
  plt.xlim([0, T*N])
  plt.ylim([-0.5, N - 0.5])
  plt.xlabel('Time step', fontsize=12)
  plt.ylabel('Neuron ID', fontsize=12)
#   plt.title('Ground truth: {:d}'.format(target_label))
  new_list = range(0,N,1)
  plt.yticks(new_list)      

#   if init_T == 0:
#     plt.title('classes: {}'.format(change_labels))
#   else:
#     plt.title('Change from class {:d} to others {}'.format(init_label, change_labels) )


# the function plot the raster of the Poisson spike train
def raster_plot_change(spike_trains, init_label, change_labels, init_T):
  """
  Generates poisson trains

  Args:
    spike_train : binary spike trains, with shape (B, N, Lt)
    n           : number of neurons

  Returns:
    Raster plot of the spike train
  """

  # find the number of all the spike trains
  B, N, T = spike_trains.shape
  time_sequence = np.arange(0, T)
  # n should smaller than N:
#   if n > N:
#     print('The number n exceeds the size of spike trains')
#     print('The number n is set to be the size of spike trains')
#     n = N
  
  # plot rater
  b = 0
  color = iter(cm.rainbow(np.linspace(0, 1, B)))
  while b < B:
    c = next(color)
    i = 0
    while i < N:
        if spike_trains[b, i, :].sum() > 0.:
            t_sp = time_sequence[spike_trains[b, i, :]>0.5]
            plt.plot(t_sp, i * np.ones(len(t_sp)), '|', ms=10, markeredgewidth=2, c=c, alpha=0.7)
        i += 1
    # print(b, spike_trains[b, :, :].sum(-1))
    # plt.show()
    b += 1
  
  change_Y = np.linspace(-0.5, N + 0.5)
  change_X = init_T * np.ones(change_Y.size)
  plt.plot(change_X, change_Y, 'k--')
  plt.xlim([time_sequence[0], time_sequence[-1]])
  plt.ylim([-0.5, N + 0.5])
  plt.xlabel('Time step', fontsize=12)
  plt.ylabel('Neuron ID', fontsize=12)
#   plt.title('Ground truth: {:d}'.format(target_label))
  if init_T == 0:
    plt.title('classes: {}'.format(change_labels))
  else:
    plt.title('Change from class {:d} to others {}'.format(init_label, change_labels) )




# the function plot the raster of the Poisson spike train
def raster_plot(spike_train, target_label):
  """
  Generates poisson trains

  Args:
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of neurons

  Returns:
    Raster plot of the spike train
  """

  # find the number of all the spike trains
  N, T = spike_train.shape
  time_sequence = np.arange(0, spike_train.shape[-1])
#   # n should smaller than N:
#   if n > N:
#     print('The number n exceeds the size of spike trains')
#     print('The number n is set to be the size of spike trains')
#     n = N

  # plot rater
  i = 0
  while i < N:
    if spike_train[i, :].sum() > 0.:
        t_sp = time_sequence[spike_train[i, :]>0.5]
        plt.plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=10, markeredgewidth=2)
    i += 1
  
  plt.xlim([time_sequence[0], time_sequence[-1]])
  plt.ylim([-0.5, N + 0.5])
  plt.xlabel('Time step', fontsize=12)
  plt.ylabel('Neuron ID', fontsize=12)
#   plt.title('Ground truth: {:d}'.format(target_label))
  
  plt.title('Ground truth: ' + str(target_label))


# the function plot the raster of the Poisson spike train
def raster_plot_single(spike_train, target_label, loss_mode):
  """
  Generates poisson trains

  Args:
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of neurons

  Returns:
    Raster plot of the spike train
  """

  # find the number of all the spike trains
  N, T = spike_train.shape
  time_sequence = np.arange(0, spike_train.shape[-1])
#   # n should smaller than N:
#   if n > N:
#     print('The number n exceeds the size of spike trains')
#     print('The number n is set to be the size of spike trains')
#     n = N
  fig = plt.figure(figsize=(4,2))
  
  spike_format = 'b.' if loss_mode=='firing_rate' else 'r.'
  # plot rater
  i = 0
  while i < N:
    if spike_train[i, :].sum() > 0.:
        t_sp = time_sequence[spike_train[i, :]>0.5]
        plt.plot(t_sp, i * np.ones(len(t_sp)), spike_format, ms=2, markeredgewidth=2)
    i += 1
  
  plt.xlabel('Time step', fontsize=10)
  plt.ylabel('Neuron ID', fontsize=10)
  plt.subplots_adjust(bottom=0.25)
  
  
  new_list = range(0,N,(int(N//5) if N>10 else 1))
  plt.yticks(new_list)
  
  plt.xlim([time_sequence[0], time_sequence[-1]])
  plt.ylim([-0.5, N-0.5])
  
#   plt.title('Ground truth: {:d}'.format(target_label))
  
#   plt.title('Ground truth: ' + str(target_label))
  



# the function plot the raster of the Poisson spike train
def plot_potential(input_potential, input_current, target_label):
  T = input_potential.shape[-1]
  # for t in range(T):
  

  plt.subplot(2,1,1)
  plt.title('Ground truth: {:d}'.format(target_label))
  plt.plot(np.arange(0,T), input_potential, ms=10, markeredgewidth=2)
  plt.xlim([0, T])
  plt.ylim([input_potential.min(), input_potential.max()])
  plt.xlabel('Time step', fontsize=12)
  plt.ylabel('PSP', fontsize=12)
  
  plt.subplot(2,1,2)
  plt.plot(np.arange(0,T), input_current, ms=10, markeredgewidth=2)
  plt.xlim([0, T])
  plt.ylim([input_current.min(), input_current.max()])
  plt.xlabel('Time step', fontsize=12)
  plt.ylabel('PSC', fontsize=12)
  
  