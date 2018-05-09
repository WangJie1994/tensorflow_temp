# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:31:15 2018

@author: wangj
"""

import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#create tensorflow structure

