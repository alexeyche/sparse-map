
import numpy as np
from util import *
from scipy import integrate


def relu(x):
    return np.maximum(x, 0.0)

def normf(F, axis=0):
    return F/np.linalg.norm(F, axis=axis)


def integrate_euler(dXdt, X0, Tvec, dt):
	X = X0.copy()
	Xseq = np.zeros((len(Tvec), len(X)))
	for ti, t in enumerate(Tvec):
		X += dt * dXdt(X, t)
		Xseq[ti] = X.copy()
	return Xseq

batch_size = 1
layer_size = 2
filter_len = 1

# setup

Tmax = 500.0
dt = 1.0
Tsize = int(Tmax/dt)
Tvec = np.linspace(0, Tmax, Tsize)

## input signal

freq = np.asarray([2.0, 2.0])
shift = np.asarray([0.0, np.pi/2.0])

input_signal = lambda t: np.sin(2.0 * np.pi * freq * t - shift)


# parameters

gain = np.asarray([10.0, 0.0001])
threshold = 0.01
feedback_gain = np.dot(gain.T, gain) - np.eye(layer_size)
tau = 5.0


f = lambda x: relu(x - threshold)


def dXdt(X, t=0):
	return (- X + gain * input_signal(t) - np.dot(f(X), feedback_gain))/tau

X0 = np.zeros((layer_size,))

X, infodict = integrate.odeint(dXdt, X0, Tvec, full_output=True)
X_euler = integrate_euler(dXdt, X0, Tvec, dt)

