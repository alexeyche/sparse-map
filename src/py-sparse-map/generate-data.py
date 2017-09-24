
import numpy as np
from config import Config, dictionarize
from util import *
import yaml
from os.path import join as pj
import env
import struct
import io_functions as io

c = Config()

np.random.seed(1)

c.batch_size = 1
c.input_size = 10
c.layer_size = 50
c.filter_size = 25

c.seq_size = 10



x_v = np.zeros((c.seq_size, c.batch_size, c.input_size))

for bi in xrange(c.batch_size):
    for si in xrange(c.seq_size):
        for ni in xrange(c.input_size):
            if np.random.random() < 0.05: #0.001:
                x_v[si, bi, ni] = 1.0


yaml.dump(dictionarize(c), open(env.yaml_config, "w"))


io.write_tensor(x_v, pj(env.work_dir, "input_data.bin")) 
