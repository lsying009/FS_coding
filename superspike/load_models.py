import torch
import torch.nn as nn

from src.models import SpikeDVSGestureCNN, SpikeFCNet, SpikeDVSPlaneCNN


def load_pt_file(model, path_to_pt, best_model=False):
    load_epoch = 0
    load_acc = 0
    checkpoint = torch.load(path_to_pt, map_location='cuda:0')
    if best_model and 'best_model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['best_model_state_dict'], strict=True)
        
        if 'best_epoch' in checkpoint.keys():
            load_epoch = checkpoint['best_epoch']
        if 'best_acc' in checkpoint.keys():
            load_acc = checkpoint['best_acc']
    elif 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        assert ('best_model_state_dict' not in checkpoint.keys() and 'model_state_dict' not in checkpoint.keys()), 'No model_state_dict!'

    if 'epoch' in checkpoint.keys() and load_epoch==0:
        load_epoch = checkpoint['epoch']
    
    if 'acc' in checkpoint.keys() and load_acc==0:
        load_acc = checkpoint['acc']
        
    return model, load_epoch, load_acc


def load_model(args, sim_params, device, test=False):
    input_size = sim_params['dataset']['input_size']
    num_classes = sim_params['dataset']['num_classes']
    print(sim_params['loss']['loss_mode'])
    if sim_params['dataset']['name'] == 'dvsplane':
        model = SpikeDVSPlaneCNN(input_size=input_size, out_size=num_classes, \
            neuron_params=sim_params['neuron_params1'], neuron_fc_params=sim_params['neuron_params2'], \
                loss_params=sim_params['loss'], device=device, test=test)
    elif sim_params['dataset']['name'] in ['dvsgesture']:
        model = SpikeDVSGestureCNN(input_size=input_size, out_size=num_classes,  \
            neuron_params=sim_params['neuron_params1'], neuron_fc_params=sim_params['neuron_params2'], \
                loss_params=sim_params['loss'], device=device, test=test)
    elif sim_params['dataset']['name'] in ['shd', 'ntidigits']:
        model = SpikeFCNet(input_size=input_size, out_size=num_classes, hidden_size=sim_params['network']['hidden_size'], \
            neuron_params1=sim_params['neuron_params1'], neuron_params2=sim_params['neuron_params2'], loss_params=sim_params['loss'], device=device, \
                test=test)
    print(sim_params['neuron_params1'])
    print(sim_params['neuron_params2'])
    
    model = model.to(device)
    

    load_epoch = 0
    load_acc = 0
    if args.load_pt_file is not None:
        model, load_epoch, load_acc = load_pt_file(model, args.load_pt_file, best_model=args.best_model)

    if args.distributed:
        model = nn.DataParallel(model, device_ids=None)
    
    print('load_epoch: ', load_epoch)
    print(sim_params)
    return model, load_epoch, load_acc


