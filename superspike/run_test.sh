#!\bin\bash

dataset_name='shd'
neuron_type='culif'
load_pt_file=../models/shd_fs0.2_tau60_culif.pt

dataset_name='dvsgesture'
neuron_type='culif'
load_pt_file=../models/dvsgesture_fs0.1_tau60_culif.pt

dataset_name='dvsplane'
neuron_type='culif'
load_pt_file=../models/dvsplane_fs0.1_tau40_culif.pt


dataset_name='ntidigits'
neuron_type='culif'
load_pt_file=../models/ntidigits_fs0.1_tau40_culif.pt


if [ $dataset_name == 'dvsgesture' ]
then
    path_to_yaml='../configs/culif_dvsgesture.yaml'
    chunck_size=250
    test_batch_size=4
elif [ $dataset_name == 'shd' ]
then
    path_to_yaml='../configs/culif_shd.yaml'
    chunck_size=100
    test_batch_size=256
elif [ $dataset_name == 'dvsplane' ]
then
    path_to_yaml='../configs/culif_dvsplane.yaml'
    chunck_size=135
    test_batch_size=16
elif [ $dataset_name == 'ntidigits' ]
then
    path_to_yaml='../configs/culif_ntidigits.yaml'
    chunck_size=250 #250
    test_batch_size=256
fi


python test.py \
--path_to_yaml $path_to_yaml \
--test-batch-size $test_batch_size \
--T $chunck_size \
--T_empty 0 \
--load_pt_file $load_pt_file \
--best_model \



python test.py \
--path_to_yaml $path_to_yaml \
--T $chunck_size \
--T_empty 0 \
--test-batch-size 20 \
--load_pt_file $load_pt_file \
--test_single_delay \
--best_model \

