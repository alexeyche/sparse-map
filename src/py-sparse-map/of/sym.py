
import sympy as sp


x, y, W = sp.symbols("x y W")
f = sp.Function("f")

lx = 0.5*(x - f(y) * W) ** 2


print "dlx/dy =", lx.diff(y)
print "dlx/dW =", lx.diff(W)


ly = 0.5*(y - f(x * W) ) ** 2

print 
print "dly/dy =", ly.diff(y)
print "dly/dW =", ly.diff(W)
