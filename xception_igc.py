# -*- coding: utf-8 -*-
"""
Xception network, for images with size 299x299

Reference:
François Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/abs/1610.02357
"""
import mxnet as mx


def separable_conv(data, n_in_ch, n_out_ch, kernel, pad, name, depth_mult=1):
    dw_out = mx.sym.Convolution(data=data, num_filter=n_in_ch, pad=pad, kernel=kernel,
                                no_bias=True, num_group=n_in_ch, name=name+'_depthwise_kernel')
    #  pointwise convolution
    pw_out = mx.sym.Convolution(dw_out, num_filter=n_out_ch, kernel=(1, 1),
                                no_bias=True, name=name+'_pointwise_kernel')
    return pw_out

def igc_conv(data, n_in_ch, n_out_ch, kernel, pad, name, depth_mult=1):
    primary_out = mx.sym.GroupSeparableConvolution(data=data, num_filter=n_in_ch, pad=pad, kernel=kernel,
                                no_bias=True, num_group=n_in_ch/2, name=name+'_primary_groupconv')
    #  pointwise convolution
    primary_reorder = mx.sym.Reorder(data = primary_out, branch_factor=n_in_ch/2, name = name+'_permutation')
    secondary_out = mx.sym.Convolution(primary_reorder, num_filter=n_out_ch, kernel=(1, 1),
                                no_bias=True, num_grup=2, name=name+'_secondary_groupconv')
    secondary_reorder = mx.sym.Reorder(data = secondary_out, branch_factor=2, name = name+'_permutation_back')
    return secondary_reorder


#def get_xception_symbol():
#def get_symbol():
def get_symbol(num_classes=1000, **kwargs):
    data = mx.sym.Variable('data')
    b1 = mx.sym.Convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2),
                            no_bias=True, name='block1_conv1')
    b1 = mx.sym.BatchNorm(b1, name='block1_conv1_bn')
    b1 = mx.sym.Activation(b1, act_type='relu', name='block1_conv1_act')
    b1 = mx.sym.Convolution(b1, num_filter=64, kernel=(3, 3), stride=(1, 1),
                            no_bias=True, name='block1_conv2')
    b1 = mx.sym.BatchNorm(b1, name='block1_conv2_bn')
    b1 = mx.sym.Activation(b1, act_type='relu', name='block1_conv2_act')

    # block 2
    rs2 = mx.sym.Convolution(b1, num_filter=160, kernel=(1, 1), stride=(2, 2),
                             no_bias=True, name='convolution2d_1')
    rs2 = mx.sym.BatchNorm(rs2, name='batchnormalization_1')

    b2 = igc_conv(b1, 64, 160, (3, 3), (1, 1), 'block2_sepconv1')
    b2 = mx.sym.BatchNorm(b2, name='block2_sepconv1_bn')
    b2 = mx.sym.Activation(b2, act_type='relu', name='block2_sepconv1_act')
    b2 = igc_conv(b2, 160, 160, (3, 3), (1, 1), 'block2_sepconv2')
    b2 = mx.sym.BatchNorm(b2, name='block2_sepconv2_bn')

    b2 = mx.sym.Pad(b2, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
    b2 = mx.sym.Pooling(b2, kernel=(3, 3), stride=(2, 2), pool_type='max',
                        pooling_convention='full', name='block2_pool')
    b2 = b2 + rs2

    # block 3
    rs3 = mx.sym.Convolution(b2, num_filter=320, kernel=(1, 1), stride=(2, 2),
                             no_bias=True, name='convolution2d_2')
    rs3 = mx.sym.BatchNorm(rs3, name='batchnormalization_2')

    b3 = mx.sym.Activation(b2, act_type='relu', name='block3_sepconv1_act')
    b3 = igc_conv(b3, 160, 320, (3, 3), (1, 1), 'block3_sepconv1')
    b3 = mx.sym.BatchNorm(b3, name='block3_sepconv1_bn')
    b3 = mx.sym.Activation(b3, act_type='relu', name='block3_sepconv2_act')
    b3 = igc_conv(b3, 160, 320, (3, 3), (1, 1), 'block3_sepconv2')
    b3 = mx.sym.BatchNorm(b3, name='block3_sepconv2_bn')

    b3 = mx.sym.Pooling(b3, kernel=(3, 3), stride=(2, 2), pool_type='max',
                        pooling_convention='full', name='block3_pool')
    b3 = b3 + rs3

    # block 4
    rs4 = mx.sym.Convolution(b3, num_filter=960, kernel=(1, 1), stride=(2, 2),
                             no_bias=True, name='convolution2d_3')
    rs4 = mx.sym.BatchNorm(rs4, name='batchnormalization_3')

    b4 = mx.sym.Activation(b3, act_type='relu', name='block4_sepconv1_act')
    b4 = igc_conv(b4, 320, 960, (3, 3), (1, 1), 'block4_sepconv1')
    b4 = mx.sym.BatchNorm(b4, name='block4_sepconv1_bn')
    b4 = mx.sym.Activation(b4, act_type='relu', name='block4_sepconv2_act')
    b4 = igc_conv(b4, 960, 960, (3, 3), (1, 1), 'block4_sepconv2')
    b4 = mx.sym.BatchNorm(b4, name='block4_sepconv2_bn')

    b4 = mx.sym.Pad(b4, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
    b4 = mx.sym.Pooling(b4, kernel=(3, 3), stride=(2, 2), pool_type='max',
                        pooling_convention='full', name='block4_pool')
    b4 = b4 + rs4

    b = b4
    for i in range(8):
        residual = b
        prefix = 'block' + str(i + 5)

        b = mx.sym.Activation(b, act_type='relu', name=prefix+'_sepconv1_act')
        b = igc_conv(b, 960, 960, (3, 3), (1, 1), prefix+'_sepconv1')
        b = mx.sym.BatchNorm(b, name=prefix+'_sepconv1_bn')
        b = mx.sym.Activation(b, act_type='relu', name=prefix+'_sepconv2_act')
        b = igc_conv(b, 960, 960, (3, 3), (1, 1), prefix+'_sepconv2')
        b = mx.sym.BatchNorm(b, name=prefix+'_sepconv2_bn')
        b = mx.sym.Activation(b, act_type='relu', name=prefix+'_sepconv3_act')
        b = igc_conv(b, 960, 960, (3, 3), (1, 1), prefix+'_sepconv3')
        b = mx.sym.BatchNorm(b, name=prefix+'_sepconv3_bn')

        b = b + residual

    rs5 = mx.sym.Convolution(b, num_filter=1280, kernel=(1, 1), stride=(2, 2),
                             no_bias=True, name='convolution2d_4')
    rs5 = mx.sym.BatchNorm(rs5, name='batchnormalization_4')

    b13 = mx.sym.Activation(b, act_type='relu', name='block13_sepconv1_act')
    b13 = igc_conv(b13, 960, 960, (3, 3), (1, 1), 'block13_sepconv1')
    b13 = mx.sym.BatchNorm(b13, name='block13_sepconv1_bn')
    b13 = mx.sym.Activation(b13, act_type='relu', name='block13_sepconv2_act')
    b13 = igc_conv(b13, 960, 1280, (3, 3), (1, 1), 'block13_sepconv2')
    b13 = mx.sym.BatchNorm(b13, name='block13_sepconv2_bn')

    b13 = mx.sym.Pad(b13, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
    b13 = mx.sym.Pooling(b13, kernel=(3, 3), stride=(2, 2), pool_type='max',
                         name='block13_pool')
    b13 = b13 + rs5

    b14 = igc_conv(b13, 1280, 1920, (3, 3), (1, 1), 'block14_sepconv1')
    b14 = mx.sym.BatchNorm(b14, name='block14_sepconv1_bn')
    b14 = mx.sym.Activation(b14, act_type='relu', name='block14_sepconv1_act')
    b14 = igc_conv(b14, 1920, 2560, (3, 3), (1, 1), 'block14_sepconv2')
    b14 = mx.sym.BatchNorm(b14, name='block14_sepconv2_bn')
    b14 = mx.sym.Activation(b14, act_type='relu', name='block14_sepconv2_act')

    pool = mx.sym.Pooling(b14, kernel=(10, 10), global_pool=True,
                          pool_type='avg', name='global_pool')
    dropout = mx.symbol.Dropout(data=pool, p=0.5)
    fc = mx.symbol.FullyConnected(data=dropout, num_hidden=num_classes,
                                  name='predictions')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

    return softmax
