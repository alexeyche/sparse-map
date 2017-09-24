import io
import env
from os.path import join as pj
import numpy as np


m = io.read_tensor(pj(env.work_dir, "input_data.bin"))
