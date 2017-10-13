
import numpy as np
from config import Config, dictionarize
from util import *
import yaml
from os.path import join as pj
import setup
import struct
import io_functions as io

c = Config()

np.random.seed(1)

c.batch_size = 1
c.input_size = 5
c.layer_size = 50
c.filter_size = 1
c.tau = 5.0
c["lambda"] = 0.05
c.seq_size = 100


c.files.activation = "/artefacts/activation.bin"
c.files.membrane = "/artefacts/membrane.bin"
c.files.F = "/artefacts/F.bin"
c.files.Fc = "/artefacts/Fc.bin"



x_v = np.zeros((c.seq_size, c.batch_size, c.input_size))

for bi in xrange(c.batch_size):
    for si in xrange(0, c.seq_size, 1):
        x_v[si, bi, si % c.input_size] = 1.0

    # for si in xrange(c.seq_size):
    #     for ni in xrange(c.input_size):
    #         if np.random.random() < 0.001:
    #             x_v[si, bi, ni] = 1.0


yaml.dump(dictionarize(c), open(setup.yaml_config, "w"))


io.write_tensor(x_v, pj(setup.work_dir, "input_data.bin")) 
