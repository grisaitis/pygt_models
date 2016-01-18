from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt

# Create the network we want
netconf = pygt.netgen.NetConf()
netconf.ignore_conv_buffer = True
netconf.use_batchnorm = False
netconf.dropout = 0.0

netconf.fmap_start = 20
netconf.fmap_inc = 1
netconf.fmap_dec = 1
netconf.unet_depth = 3
netconf.unet_downsampling_strategy = [[3,3,3],[3,3,3],[1,1,1]]
netconf.input_shape = [178, 178, 178]
netconf.output_shape = [38, 38, 38]

print ('Input shape: %s' % netconf.input_shape)
print ('Output shape: %s' % netconf.output_shape)
print ('Feature maps: %s' % netconf.fmap_start)

netconf.loss_function = "euclid"
train_net_conf_euclid, test_net_conf = pygt.netgen.create_nets(netconf)
netconf.loss_function = "malis"
train_net_conf_malis, test_net_conf = pygt.netgen.create_nets(netconf)

with open('net_train_euclid.prototxt', 'w') as f:
    print(train_net_conf_euclid, file=f)
with open('net_train_malis.prototxt', 'w') as f:
    print(train_net_conf_malis, file=f)
with open('net_test.prototxt', 'w') as f:
    print(test_net_conf, file=f)
