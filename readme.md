# First-Spike Coding of SNNs

[Siying Liu (siying.liu20@imperial.ac.uk)](https://www.imperial.ac.uk/people/siying.liu20), [Vincent C. H. Leung](https://scholar.google.com/citations?user=UUlylYYAAAAJ&hl=en&oi=sra), and [Pier Luigi Dragotti](https://www.commsp.ee.ic.ac.uk/~pld/).

This repository is the official PyTorch implementation of the paper [First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures](https://www.frontiersin.org/articles/10.3389/fnins.2023.1266003/full).

![FS arch](https://www.frontiersin.org/files/Articles/1266003/fnins-17-1266003-HTML/image_m/fnins-17-1266003-g002.jpg)

## Implementation
There are two implementation versions of SNNs, based on [SuperSpike](https://arxiv.org/abs/1705.11146) and [SpikingJelly](https://github.com/fangwei123456/spikingjelly) under folders named ```superspike``` and ```spkjelly```, respectively. Please install ```spikingjelly==0.0.0.0.14``` before use.

In ```superspike```, the backpropagation of neurons is implemented by PyTorch Autograd, training speed is relatively slow. Only Current-based LIF (CuLIF) neuron with fixed or trainable time constants are implemented.

In ```spkjelly```, the FP and BP of neurons are accelerated when using ```backend='cupy'``` under ```step_mode='m'```.   We implemented the cupy backend for three types of spiking neurons, including CuLIF, Parametric CuLIF (PCuLIF) and [Adaptive LIF (AdLIF)](https://www.frontiersin.org/articles/10.3389/fnins.2022.865897/full) neurons. We recommend using this implementation, since the training is much faster and models with AdLIF for SHD and NTIDIGITS achieve the best performance.


## Datasets
Four datasets are trained and tested, including audio datasets [SHD](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), [N-TIDIGITIS](https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit#heading=h.sbnu5gtazqjq) and visual datasets [DVSGesture](https://research.ibm.com/interactive/dvsgesture/), [DVSPlane](http://greg-cohen.com/datasets/dvs-planes/). These data are transformed into ```.hdf5``` format, which can be downloaded [here](https://drive.google.com/drive/folders/10-9ezGNdfZJKFKDDYQge_vPzBZIOSm54?usp=sharing). Put them in the ```datasets``` folder.


## Training
Run ```bash run.sh``` to train. Parameter settings can be found in ```.yaml``` under ```configs```. Parameters can also be specified by args. Please ```cd superspike``` or ```cd spkjelly``` before run ```bash run.sh```.

An example of training.
    
    cd spikingjelly
    path_to_yaml='../configs/adlif_shd.yaml'
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
    --hidden_size 256 256 \
    --FS 0.2 16 200 \
    --neuron1 5. 5. 0.5 \
    --neuron2 60. 60. 10 \
    --treg 0.01 0.02 \

## Testing
Each model can be evaluated with FS/FR coding. Run ```bash run_test.sh```, check the FR/FS accuracy, spike count, time delay, spike pattern, etc.

An example of testing.

    python test.py \
    --path_to_yaml $path_to_yaml \
    --test-batch-size $test_batch_size \
    --T $chunck_size \
    --T_empty 0 \
    --load_pt_file $load_pt_file \
    --best_model \


## Citation
If you use any of this code, please cite the publications as follows:
```bibtex
@ARTICLE{10.3389/fnins.2023.1266003,
  AUTHOR={Liu, Siying and Leung, Vincent C. H. and Dragotti, Pier Luigi},   
  TITLE={First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures},      
  JOURNAL={Frontiers in Neuroscience},      
  VOLUME={17},           
  YEAR={2023},      
  URL={https://www.frontiersin.org/articles/10.3389/fnins.2023.1266003},       
  DOI={10.3389/fnins.2023.1266003},      
  ISSN={1662-453X},   
}
```