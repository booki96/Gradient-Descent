import numpy as np
import math, matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from sympy import symbols, diff


class gd_cw_2d:
    def __init__(self, fn_loss, fn_grad):
        self.fn_loss = fn_loss
        self.fn_grad = fn_grad

    #function for plain vanilla gradient descent
    def vanilla(self, x1_init, x2_init, n_iter, eta, tol):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        x = [x1_init,x2_init]
        
        loss_path = []
        x_path = np.zeros([n_iter+1,2])
        
        x_path[0,:] = x
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        
        g = self.fn_grad(x[0],x[1])
        grads = []
        for i in range(n_iter):
            grad = np.ndarray.tolist(g)
            grad = [float(j) for j in grad]
            
            if np.all(np.abs(g)) < tol or np.any(np.isnan(grad)):
                break
            g = self.fn_grad(x[0],x[1])
            x += -eta*g
            x_path[i+1,:] = x
            
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)
            grads.append(g)

        if np.any(np.isnan(grad)):
            print('Vanilla Exploded')
        elif np.any(np.abs(g)) > tol:
            print('Vanilla did not converge')
        
        self.loss_path = loss_path
        self.x_path = x_path[0:i+1]
        self.loss_fn_min = loss_this
        self.x_at_min = x
        self.n_iter = i
        self.grad = grads

    # function for gradient descent with momentum
    def momentum(self, x1_init, x2_init, n_iter, eta, tol, alpha):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.alpha = alpha
        x = [x1_init,x2_init]
        
        loss_path = []
        x_path = np.zeros([n_iter+1,2])
        
        x_path[0,:] = x
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])

        nu = 0
        for i in range(n_iter):
            g = self.fn_grad(x[0],x[1])
            grad = np.ndarray.tolist(g)
            grad = [float(j) for j in grad]
            
            if np.all(np.abs(g)) < tol or np.any(np.isnan(grad)):
                break
            nu = alpha * nu + eta * g
            x += -nu
            
            x_path[i+1,:] = x
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)
            
        if np.any(np.isnan(grad)):
            print('Momentum Exploded')
        elif np.any(np.abs(g)) > tol:
            print('Momentum Did not converge')
        
        self.loss_path = loss_path
        self.x_path = x_path[0:i+1]
        self.loss_fn_min = loss_this
        self.x_at_min = x

    # function for gd with nesterov
    def nag(self, x1_init, x2_init, n_iter, eta, tol, alpha):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.alpha = alpha
        x = [x1_init,x2_init]

        loss_path = []
        x_path = np.zeros([n_iter+1,2])

        x_path[0,:] = x
        loss_this = self.fn_loss(x[0],x[1])
        loss_path.append(loss_this)
        g = self.fn_grad(x[0],x[1])
        g_mag= sum(np.square(x))

        nu = np.array([0,0])
        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g = self.fn_grad(x[0]-mu*nu[0],x[1]-mu*nu[1])
            grad = np.ndarray.tolist(g)
            grad = [float(j) for j in grad]
            
            if np.all(np.abs(g)) < tol or np.any(np.isnan(grad)):
                break
            
            nu = alpha * nu + eta * g
            x += -nu
            
            x_path[i+1,:] = x
            loss_this = self.fn_loss(x[0],x[1])
            loss_path.append(loss_this)
            
        if np.any(np.isnan(grad)):
            print('NAG Exploded')
        elif np.all(np.abs(g)) > tol:
            print('NAG Did not converge')
            
        self.loss_path = loss_path
        self.x_path = x_path[0:i+1]
        self.loss_fn_min = loss_this
        self.x_at_min = x

# loss function  
def fn_loss(x,y):
    loss = (x+2*y-7)**2 + (2*x+y-5)**2
    return loss

# function for calculating gradient
def fn_grad(x0,y0):
    x,y = symbols('x y',real=True)
    loss = (x+2*y-7)**2 + (2*x+y-5)**2
    d1 = diff(loss,x)
    d2 = diff(loss,y)
    g1 = d1.evalf(subs={x: x0,y: y0})
    g2 = d2.evalf(subs={x: x0,y: y0})
    g = np.array([g1,g2])
    return g

# function for plotting loss function and path of gd
def plot(path):
    ax = plt.axes(projection='3d')

    x = np.linspace(-10,10)
    y = np.linspace(-10,10)

    X, Y = np.meshgrid(x,y)


    Z = fn_loss(X,Y)

    ax.plot_surface(X, Y, Z, alpha=0.7)

    x1 = path[:,0]
    y1 = path[:,1]
    z1 = fn_loss(x1,y1)

    ax.scatter(x1,y1,z1,c='b')

    ax.set_xlabel('X',fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('f(X,Y)',fontsize=15, rotation=0)
    plt.show()

def convergence(n_iter,loss_path):
    x = [i for i in range(n_iter)]
    y = loss_path

    plt.scatter(x,y)
    plt.xlabel('Steps', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    

# initalise the class
solver = gd_cw_2d(fn_loss=fn_loss, fn_grad=fn_grad)

# plain vanilla gradient descent & input parameters 
solver.vanilla(x1_init = -8, x2_init = -8, n_iter = 10000,
               eta = .1, tol = 1e-10)
vanilla_path = solver.x_path
vanilla_min = solver.x_at_min.astype(np.double)
vanilla_loss = solver.loss_path
grad = solver.grad

print('The point which which generates the minimum (vanilla) is: ', vanilla_min)
plot(vanilla_path)
convergence(len(vanilla_loss),vanilla_loss)

# gradient descent with momentum & input parameters
solver.momentum(x1_init = -8, x2_init = -8, n_iter = 10000,
                eta = 0.1, tol = 1e-10, alpha = .5)
momentum_path = solver.x_path
momentum_min = solver.x_at_min.astype(np.double)
momentum_loss = solver.loss_path

print('The point which generates the minimum (momentum) is: ', momentum_min)
plot(momentum_path)

# gradient descent with nesterov & input parameters
solver.nag(x1_init = -8, x2_init = -8, n_iter = 10000,
           eta = 0.01, tol = 1e-10, alpha = .5)
nag_path = solver.x_path
nag_min = solver.x_at_min.astype(np.double)
nag_loss = solver.loss_path

print('The point which generates the minimum (nag) is: ', nag_min)
plot(nag_path)


