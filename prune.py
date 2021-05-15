import torch

from network import VGG
from train import train_network
from preresnet import resnet

def prune_network(args, network=None):
    resnet_prune_layer = 1
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")

    if args.net == 'resnet50' and network is None:
        network = resnet()
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])
    elif network is None:
        network = VGG(args.net, args.data_set)
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])

    # prune network
    if args.net == 'resnet50':
        if resnet_prune_layer == 1:
            network = prune_resnet_1(network, args.prune_layers, args.independent_prune_flag)
        if resnet_prune_layer == 2:
            network = prune_resnet_2(network, args.prune_layers, args.independent_prune_flag)
        if resnet_prune_layer == 3:
            network = prune_resnet_3(network, args.prune_layers, args.independent_prune_flag)
        
    else:
        network = prune_step(network, args.prune_layers, args.prune_channels, args.independent_prune_flag)
    network = network.to(device)
    print("-*-"*10 + "\n\tPrune network\n" + "-*-"*10)
    print(network)

    if args.retrain_flag:
        # update arguments for retraining pruned network
        args.epoch = args.retrain_epoch
        args.lr = args.retrain_lr
        args.lr_milestone = None # don't decay learning rate

        network = train_network(args, network)
    
    return network

def prune_step(network, prune_layers, prune_channels, independent_prune_flag):
    network = network.cpu()

    count = 0 # count for indexing 'prune_channels'
    conv_count = 1 # conv count for 'indexing_prune_layers'
    dim = 0 # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None # residue is need to prune by 'independent strategy'
    for i in range(len(network.features)):
        print(network.features[i])
        if isinstance(network.features[i], torch.nn.Conv2d):
            if dim == 1:
                # prune next layer's filter in dim=1
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1

            if 'conv%d'%conv_count in prune_layers: 
                # identify channels to remove        
                channel_index = get_channel_index(network.features[i].weight.data, prune_channels[count], residue)
                # prune current layer's filter in dim=0
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1

        # prune bn
        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    # update to check last conv layer pruned
    if 'conv16' in prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network

def prune_resnet_1(net, prune_layers, independent_prune_flag):
    # init
    arg_index = 1
    residue = None
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    prune_rate = [0.3,0.3,0.3,0.3]

    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            if 'block%d'%arg_index in prune_layers:
                # identify channels to remove
                remove_channels = get_channel_index(layers[layer_index][block_index].conv1.weight.data,
                                                    int(round(layers[layer_index][block_index].conv1.out_channels * prune_rate[layer_index])), residue)
                # prune current layer's filter in dim=0
                layers[layer_index][block_index].conv1 = get_new_conv(layers[layer_index][block_index].conv1,0,
                                                                    remove_channels, independent_prune_flag)
                # prune next layer's filter in dim=1
                layers[layer_index][block_index].conv2, residue = get_new_conv(
                layers[layer_index][block_index].conv2, 1, remove_channels, independent_prune_flag)
                residue = None

                # prune bn
                layers[layer_index][block_index].bn2 = get_new_norm(layers[layer_index][block_index].bn2,
                                                                remove_channels)
            arg_index += 1
    #print(arg_index)        
    return net

def prune_resnet_2(net, prune_layers, independent_prune_flag):
    # init
    arg_index = 1
    residue = None
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    prune_rate = [0.3,0.3,0.3,0.3]

    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            if 'block%d'%arg_index in prune_layers:
                # identify channels to remove
                remove_channels = get_channel_index(layers[layer_index][block_index].conv2.weight.data,
                                                    int(round(layers[layer_index][block_index].conv2.out_channels * prune_rate[layer_index])), residue)
                # prune current layer's filter in dim=0
                layers[layer_index][block_index].conv2 = get_new_conv(layers[layer_index][block_index].conv2,0,
                                                                    remove_channels, independent_prune_flag)
                # prune next layer's filter in dim=1
                layers[layer_index][block_index].conv3, residue = get_new_conv(
                layers[layer_index][block_index].conv3, 1, remove_channels, independent_prune_flag)
                residue = None

                # prune bn
                layers[layer_index][block_index].bn3 = get_new_norm(layers[layer_index][block_index].bn3,
                                                                remove_channels)
            arg_index += 1
    #print(arg_index)        
    return net

def prune_resnet_3(net, prune_layers, independent_prune_flag):
    # init
    arg_index = 1
    residue = None
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    prune_rate = [0.3,0.3,0.3,0.3]
    last_prune = False

    for layer_index in range(len(layers)):
        for block_index in range(len(layers[layer_index])):
            if last_prune:
                # if current stage's shortcut has been pruned, subsequent residual block's first bn layer and conv1 layer should also been pruned
                layers[layer_index][block_index].bn1 = get_new_norm(layers[layer_index][block_index].bn1,
                                                                remove_channels)
                layers[layer_index][block_index].conv1, residue = get_new_conv(layers[layer_index][block_index].conv1, 1, remove_channels, independent_prune_flag)
                if layers[layer_index][block_index].downsample is not None:
                    # prune next stage's shortcut in dim=1
                    layers[layer_index][block_index].downsample[0], residue = get_new_conv(layers[layer_index][block_index].downsample[0], 1, remove_channels, independent_prune_flag)

            if layers[layer_index][block_index].downsample is not None:
                print(arg_index)
                if 'block%d'%arg_index in prune_layers:
                    # identify channels to remove
                    remove_channels = get_channel_index(layers[layer_index][block_index].downsample[0].weight.data,
                                                        int(round(layers[layer_index][block_index].downsample[0].out_channels * prune_rate[layer_index])), residue)
                    # prune shortcut's filter in dim=0
                    layers[layer_index][block_index].downsample[0] = get_new_conv(layers[layer_index][block_index].downsample[0],0,
                                                                        remove_channels, independent_prune_flag)
                    last_prune = True
                    residue = None
                else:
                    # to identify that current stage's shortcut has not been pruned
                    last_prune = False
                    residue = None

            if last_prune:
                # if current stage's shortcut has been pruned, all residual block's conv3 layer should also been pruned
                # the third layer of the residual block is pruned with the same filter index as selected by the pruning of the shortcut
                layers[layer_index][block_index].conv3 = get_new_conv(
                layers[layer_index][block_index].conv3, 0, remove_channels, independent_prune_flag)
            
            arg_index += 1
    
    # update to check last conv layer pruned
    if 'block14' in prune_layers:
        net.classifier = get_new_linear(net.classifier, remove_channels)
        net.bn = get_new_norm(net.bn, remove_channels)
    return net

def get_channel_index(kernel, num_elimination, residue=None):
    # get cadidate channel index for pruning
    ## 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    
    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()

def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))
    
    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor

def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        if conv.bias is not None:
            new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        
        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        return new_conv, residue

def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)
        
    return new_norm

def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                out_features=linear.out_features,
                                bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data
    
    return new_linear
