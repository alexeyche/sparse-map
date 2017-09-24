
import struct
import env
from os.path import join as pj
import numpy as np

def read_int(s):
	int_val, = struct.unpack('I', s.read(4))
	s.read(4)
	return int_val

def write_int(s, int_val):
	s.write(struct.pack('I', int_val))
	s.write(struct.pack('I', 0))


def read_tensor(f):
	f = open(f, "r") 
	
	shape_size = read_int(f)
	
	shape, size = [], 1
	for _ in xrange(shape_size):
		s = read_int(f)
		shape.append(s)
		size *= s

	m = np.fromfile(f, dtype=np.double, count=size)
	
	return m.reshape(tuple(shape))



def write_tensor(m, f):
	f = open(f, "w") 
	shape = m.shape

	write_int(f, len(shape))
	for s in shape:
		write_int(f, s)
	
	m.tofile(f)


if __name__ == '__main__':
	v = np.random.random((10, 23, 64)).astype(np.double)
	
	filename = "/var/tmp/123.bin"
	write_tensor(v, filename)

	v2 = read_tensor(filename)

	assert np.all(v2 == v)