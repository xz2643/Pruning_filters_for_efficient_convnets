Pruning Filters For Efficient ConvNets
==
**Pruning VGG19 and Resnet50 on CIFAR-10 Dataset**

**Reference**: [Pruning Filters For Efficient ConvNets, ICLR2017](https://arxiv.org/abs/1608.08710)

https://github.com/tyui592/Pruning_filters_for_efficient_convnets

Usage
--

### Arguments
* `--train-flag`: Train VGG or Resnet on CIFAR Dataset
* `--save-path`: Path to save results (e.g. trained_models/)
* `--load-path`: Path to load checkpoint, add 'checkpoint.pth' with `save_path`, (e.g. trained_models/checkpoint.pth)
* `--resume-flag`: Resume the training from checkpoint loaded with `load-path`
* `--prune-flag`: Prune VGG or Resnet
* `--prune-layers`: List of target convolution layers (VGG19) or residual blocks(Resnet50) for pruning (e.g. conv1 conv2/block1 block2)
* `--prune-channels`: List of number of channels for pruning the prune-layers (only for VGG19 as Resnet50 uses ratio instead)
* `--independent-prune-flag`: Prune multiple layers by independent strategy (if it is not used, then greedy approach is applied)
* `--retrain-flag`: Retrain the pruned nework
* `--retrain-epoch`: Number of epoch for retraining pruned network
* `--retrain-lr`: Retrain learning rate, default=0.001

### Example Scripts

#### Train VGG19 on CIFAR-10 Dataset
```
python3 main.py --train-flag --data-set CIFAR10 --vgg vgg19_bn --save-path ./vgg_trained_models/
```

#### Prune VGG19 by 'greedy strategy'
```
python3 main.py --vgg vgg19_bn --prune-flag --load-path ./vgg_trained_models/check_point.pth --save-path ./vgg_trained_models/pruning_results/ --prune-layers conv1 conv9 conv10 conv11 conv12 conv13 conv14 conv15 conv16 --prune-channels 46 256 256 256 256 256 256 256 256
```

#### Prune VGG19 by 'independent strategy'
```
python3 main.py --vgg vgg19_bn --prune-flag --load-path ./vgg_trained_models/check_point.pth --save-path ./vgg_trained_models/pruning_results/ --prune-layers conv1 conv9 conv10 conv11 conv12 conv13 conv14 conv15 conv16 --prune-channels 46 256 256 256 256 256 256 256 256 --independent-prune-flag
```

#### Retrain the pruned VGG19 network
```
python3 main.py --vgg vgg19_bn --prune-flag --load-path ./vgg_trained_models/check_point.pth --save-path ./vgg_trained_models/pruning_results/ --prune-layers conv1 conv9 conv10 conv11 conv12 conv13 conv14 conv15 conv16 --prune-channels 46 256 256 256 256 256 256 256 256 --independent-prune-flag --retrain-flag --retrain-epoch 40 --retrain-lr 0.001
```

#### Train Resnet50 on CIFAR-10 Dataset

```
python3 main.py --train-flag --data-set CIFAR10 --vgg resnet50 --save-path ./resnet_trained_models/ --resume-flag --load-path ./resnet_trained_models/check_point.pth
```

#### Prune Resnet50 by 'greedy strategy'

```
python3 main.py  --vgg resnet50 --prune-flag --load-path ./resnet_trained_models/check_point.pth --save-path ./resnet_trained_models/pruning_results/ --prune-layers block1 block2 block5 block6 block7 block9 block10 block11 block12 block15 block16
```

#### Prune Resnet50 by 'independent strategy'

```
python3 main.py  --vgg resnet50 --prune-flag --load-path ./resnet_trained_models/check_point.pth --save-path ./resnet_trained_models/pruning_results/ --prune-layers block1 block2 block5 block6 block7 block9 block10 block11 block12 block15 block16 --independent-prune-flag
```

#### Retrain the pruned Resnet50 network

```
python3 main.py  --vgg resnet50 --prune-flag --load-path ./resnet_trained_models/check_point.pth --save-path ./resnet_trained_models/pruning_results/ --prune-layers block1 block2 block5 block6 block7 block9 block10 block11 block12 block15 block16 --independent-prune-flag --retrain-flag --retrain-epoch 40 --retrain-lr 0.001
```

