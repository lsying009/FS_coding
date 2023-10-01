# First-Spike Coding

This repository is the official PyTorch implementation of the paper [First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures](https://www.frontiersin.org/articles/10.3389/fnins.2023.1266003/abstract).

## Dataset




### Training
```cd superspike```
```cd spikingjelly```
An example of training E2V model.
    
    python train_e2v.py \
    --path_to_train_data $path_to_train_data \
    --model_name "RecNet" \
    --model_mode "cista-lstc" \
    --batch_size 1 --epochs 60 --lr 1e-4 \
    --len_sequence 15 \
    --num_bins 5 \
    --depth 5 --base_channels 64 \
    --num_events 15000 \

### Testing
```test_data_mode='real'``` for [HQF](https://timostoff.github.io/20ecnn) and [ECD](https://rpg.ifi.uzh.ch/davis_data.html) data sequences, and ```test_data_mode='upsampled'``` for simulated data sequences.
    
    python test_e2v.py \
    --path_to_test_model pretrained/RecNet_cista-lstc.pth.tar \
    --path_to_test_data data_examples/ECD \
    --reader_type 'image_reader' \
    --model_mode "cista-lstc" \
    --test_data_mode 'real' \
    --num_events 15000 \
    --test_data_name slider_depth \


## Citation
If you use any of this code, please cite the publications as follows:
```bibtex
@article{liu17first,
  title={First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures},
  author={Liu, Siying and Leung, Vincent CH and Dragotti, Pier Luigi},
  journal={Frontiers in Neuroscience},
  volume={17},
  pages={1266003},
  publisher={Frontiers}
}
```