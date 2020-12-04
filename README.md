# Hard Label Black-box Adversarial Attacks

This is code associated with the paper Hard-Label Black Box Adversarial Attacks in Low Budget Query Regimes.

Prereqs:

* Numpy
* Scipy
* Pytorch 1.4
* Torchvision 0.5
* Botorch 0.2 (see https://botorch.org/#quickstart for installation)

To rerun, e.g., the L_2 norm attack experiments against the resnet50 architecture with epsilon = 20, you can run

```
python attack.py --dset imagenet --arch resnet50 --iter 995 --eps 20.0 --dim 18 --num_attacks 1000  --optimize_acq scipy --hard_label --save
```

* For consistency, we fixed a set of 1000 ImageNet validation set images, and performed all of the experiments in our paper on this set. The indices of these images in the ImageNet validation set are contained in ```random_indices_imagenet.npy```.
