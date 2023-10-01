import argparse
import yaml
import torch

def parse_netparams(yaml_file):
    with open(yaml_file, 'r') as f:
        sim_params = yaml.safe_load(f)
    return sim_params


def create_save_name(sim_params, save_name=None, add_name=None):
    if save_name is not None:
        return save_name
    else:
        if sim_params['loss']['loss_mode'] in ['first_time']:
            loss = 'fs{}'.format(sim_params['loss']['FS']['alpha'])
        if sim_params['loss']['loss_mode'] == 'firing_rate':
            loss = 'fr{}'.format(sim_params['loss']['FR']['alpha'])
 
        create_save_name = '{}_{}_tau{:.0f}'.format(
                                   sim_params['dataset']['name'].casefold(),
                                   loss,
                                   sim_params['neuron_params2']['tau_m'],
                                   )
        if add_name is not None:
            create_save_name = create_save_name + '_{}'.format(add_name)
        return create_save_name

def load_sim_params(args, test=False):
    sim_params0 = parse_netparams(args.path_to_yaml)
    
    if not test:
        if args.batch_size:
            sim_params0['learning']['batch_size'] = args.batch_size
        if args.lr:
            sim_params0['learning']['lr'] = args.lr
        if args.weight_decay:
            sim_params0['learning']['weight_decay'] = args.weight_decay
        if args.lr_decay:
            sim_params0['learning']['lr_decay'] = args.lr_decay
        if args.epochs:
            sim_params0['learning']['epochs'] = args.epochs
        if args.clip_norm:
            sim_params0['learning']['clip_norm'] = args.clip_norm
   
    ################ update using args ################        
    if args.dataset_name:
        sim_params0['dataset']['name'] = args.dataset_name
    if args.path_to_dataset:
        sim_params0['dataset']['path_to_dataset'] = args.path_to_dataset
    if args.num_classes:
        sim_params0['dataset']['num_classes'] = args.num_classes
    
        
    if args.hidden_size:
        sim_params0['network']['hidden_size'] = args.hidden_size

    # #####################
    # # if args.is_recurrent:
    # sim_params0['network']['recurrent'] = args.is_recurrent
    # # if args.dropout:
    # sim_params0['network']['dropout'] = args.dropout
    
    if args.T:
        sim_params0['simulation']['T'] = args.T
    if args.dt:
        sim_params0['simulation']['dt'] = args.dt
    if args.T_empty:
        sim_params0['simulation']['T_empty'] = args.T_empty

    
    if args.neuron1:
        sim_params0['neuron_params1'] = dict(zip(['tau_m', 'tau_s', 'theta'], args.neuron1))
    if args.neuron2:
        sim_params0['neuron_params2'] = dict(zip(['tau_m', 'tau_s', 'theta'], args.neuron2))

    if args.loss_mode:
        sim_params0['loss']['loss_mode'] = args.loss_mode

    if args.FS:
        if len(args.FS) == 2:
            sim_params0['loss']['FS'] = dict(zip(['alpha', 'D'], args.FS))
        elif len(args.FS) == 3:
            sim_params0['loss']['FS'] = dict(zip(['alpha', 'D', 'A'], args.FS))
            # sim_params0['loss']['FS']['D'] = int(sim_params0['loss']['FS']['D'])
        else:
            sim_params0['loss']['FS']['alpha'] = args.FS[0]
            
    if args.FR:
        sim_params0['loss']['FR']['alpha'] = args.FR
        
    if args.treg:
        sim_params0['loss']['treg'] = dict(zip(['lambda', 'beta'], args.treg))

    if args.load_pt_file:
        checkpoint = torch.load(args.load_pt_file, map_location='cuda:0')
        if 'sim_params' in checkpoint.keys():
            sim_params = checkpoint['sim_params'].copy()
            sim_params['simulation'] = sim_params0['simulation']
            sim_params['learning'] = sim_params0['learning']
        if not test:
            sim_params['loss'] = sim_params0['loss'].copy()
    else:
        sim_params = sim_params0
    
    ############ compute real lenghth of sequence in the training #######
    sim_params['simulation']['input_T'] = sim_params['simulation']['T'] + sim_params['simulation']['T_empty']
    
    return sim_params


def config_common(parser):
    parser.add_argument('--path_to_yaml', type=str, help='Path to yaml config file')
    parser.add_argument('--distributed', action='store_true')
    parser.set_defaults(distributed=False)
    
    ############# dataset parameters ##############
    parser.add_argument('--path_to_dataset', type=str, help='Path to training and testing dataset')
    parser.add_argument('--dataset_name', type=str, help='DvsGesture / NTIDIGITS / NMNIST / shd / ssc / asl')
    parser.add_argument('--num_classes', type=int, help='number of classes in the dataset')

    ############## simulation parameters ################
    parser.add_argument('--dt', type=float, help='Sample time duration in microseconds')
    parser.add_argument('--T', type=int, help='Sample time length of a sequence in microseconds')
    parser.add_argument('--T_empty', type=int, help='Length of empty sequences')

    ############## model parameters ################
    parser.add_argument('--hidden_size', nargs='+', type=int, help='hidden size for FC network')

    ############## neuron paramters ################
    parser.add_argument('--neuron1', nargs=3, type=float, \
        metavar=('tau_m', 'tau_s', 'theta'), help='parameters for neuron_params 1')    
    parser.add_argument('--neuron2', nargs=3, type=float, \
        metavar=('tau_m', 'tau_s', 'theta'), help='parameters for neuron_params 2')    
    
    ############## loss parameters #################
    parser.add_argument('--loss_mode', type=str, help='firing_rate / first_time')
    parser.add_argument('--FS', nargs='+', type=float, \
        metavar=('alpha, D, A'), help='parameters for FS loss')    
    parser.add_argument('--FR', type=float, \
        help='parameters for FR loss (alpha)') 
    parser.add_argument('--treg', nargs=2, type=float, \
        metavar=('lambda', 'beta'), help='parameters for first time regularisation')   
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--best_model', dest='best_model', action='store_true')
    parser.set_defaults(best_model=False)
    return parser


def config_train():
    parser = argparse.ArgumentParser(description='Training configs')
    parser = config_common(parser)
    
    ############# training parameters #############
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay')
    parser.add_argument('--lr_decay', action='store_true',
                        help='learning rate decay=True/False')
    parser.set_defaults(lr_decay=False)
    parser.add_argument('--clip_norm', default=10, type=float, help='clip grad norm')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    ############# data io parameters #############
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--add_name', type=str, default=None)
    parser.add_argument('--load_pt_file', default=None, type=str, help='path to checkpoint file')

    return parser


def config_test():
    parser = argparse.ArgumentParser(description='Testing configs')
    
    parser = config_common(parser)
    
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--test_single_delay', action='store_true', default=False,
                        help='show the output spikes (delay) for each class')

    parser.add_argument('--load_pt_file', type=str, help='path to checkpoint file')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--show', action='store_true', default=False,
                        help='whether to show raster plot')

    return parser
