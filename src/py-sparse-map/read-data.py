import io_functions as io
import env
from os.path import join as pj
import numpy as np
from util import *

a = io.read_tensor(pj(env.work_dir, "activation.bin"))
m = io.read_tensor(pj(env.work_dir, "membrane.bin"))
F = io.read_tensor(pj(env.work_dir, "F.bin"))
Fc = io.read_tensor(pj(env.work_dir, "Fc.bin"))

# m = io.read_tensor(pj(env.work_dir, "input_data.bin"))
