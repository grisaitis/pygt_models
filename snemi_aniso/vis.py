from __future__ import print_function
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
import sys

# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), pygt_path))

# Other python modules
import math

# Load PyGreentea
import PyGreentea as pygt
import visualizer

# Load the datasets
raw_h5_fname = '/groups/turaga/home/turagas/data/SNEMI3D/train/raw.hdf5'
gt_h5_fname = '/groups/turaga/home/turagas/data/SNEMI3D/train/labels_id.hdf5'
aff_h5_fname = '/groups/turaga/home/turagas/data/SNEMI3D/train/labels_aff11.hdf5'
test_h5_fname = 'test_out_0.h5'

raw_h5f = h5py.File(raw_h5_fname,'r')
gt_h5f = h5py.File(gt_h5_fname,'r')
aff_h5f = h5py.File(aff_h5_fname,'r')
test_h5f = h5py.File(test_h5_fname,'r')

# visualizer.inspect_3D_hdf5(raw_h5f)
# visualizer.inspect_3D_hdf5(gt_h5f)
# visualizer.inspect_4D_hdf5(aff_h5f)
# visualizer.inspect_4D_hdf5(test_h5f)
visualizer.display_raw(raw_h5f['main'],50)
visualizer.display_con(test_h5f['main'],50)