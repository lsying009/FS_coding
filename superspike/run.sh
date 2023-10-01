#! \bin\bash
path_to_yaml='..configs/culif_shd.yaml'

python train.py \
--path_to_yaml $path_to_yaml \
--log-interval 5 \
--lr 1e-3 \
--batch-size 128 \
--test-batch-size 256 \
--epochs 80 \
--loss_mode 'first_time' \
--T 100 \
--T_empty 0 \
--dt 10 \
--hidden_size 256 \
--FS 0.2 16 200 \
--neuron1 5. 5. 5 \
--neuron2 60. 60. 10 \
--treg 0.01 0.02 \
--seed 2 \
--add_name 'culif' \


path_to_yaml='..configs/culif_ntidigits.yaml'

python train.py \
--path_to_yaml $path_to_yaml \
--log-interval 5 \
--lr 1e-3 \
--batch-size 128 \
--test-batch-size 512 \
--epochs 200 \
--loss_mode 'first_time' \
--hidden_size 256 256 \
--T 250 \
--T_empty 0 \
--dt 5 \
--FS 0.1 16 500 \
--neuron1 5. 5. 1 \
--neuron2 40. 40. 2 \
--treg 0.01 0.02 \
--add_name 'culif' \


path_to_yaml='..configs/culif_dvsgesture.yaml'

python train.py \
--path_to_yaml $path_to_yaml \
--log-interval 5 \
--lr 1e-4 \
--batch-size 16 \
--test-batch-size 32 \
--epochs 70 \
--T_empty 40 \
--T 120 \
--dt 10 \
--FS 0.1 4 300 \
--loss_mode 'first_time' \
--neuron1 5. 5. 0.5 \
--neuron2 60. 60. 1.0 \
--treg 0.01 0.02 \
--seed 3 \
--distributed \

path_to_yaml='..configs/culif_dvsplane.yaml'

python train.py \
--path_to_yaml $path_to_yaml \
--log-interval 5 \
--lr 3e-4 \
--batch-size 16 \
--test-batch-size 64 \
--epochs 50 \
--T_empty 0 \
--T 135 \
--dt 2 \
--FS 0.1 8 500 \
--loss_mode 'first_time' \
--neuron1 5. 5. 0.5 \
--neuron2 40. 40. 2.0 \
--treg 0.01 0.02 \
--seed 2 \
--distributed \
