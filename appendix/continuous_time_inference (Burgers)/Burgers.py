"""
@author: Maziar Raissi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        
        self.lb = lb
        self.ub = ub
    
        # Location of u training points
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        # Location of f training points
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        # Exact solution at selected boundary conditions
        self.u = u
        
        self.layers = layers

        # Coefficient in front of diffusion term
        self.nu = nu
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # TensorFlow placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Input: x coordinate for u observations
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        # Input: t coordinate for u observations
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        # Input: u observations
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        # Input: x coordinate for f evaluations
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        # Input: t coordinate for f evaluations
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        

        # u and f predictions  
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)         
        
        # Loss is defined as the mean square error for the u observations
        # and right-hand side f.
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        # Iteration counters
        self.iteration = 0

        # Setup the optimizer: L-BFGS-B
        # Limited-memory BFGS (Broyden–Fletcher–Goldfarb–Shanno)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 2000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1e-6})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Initialize weights and biases in NN
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    # Defining all the layers in the NN
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        # Rescale input between -1 and 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            # Layer is fully connected with matrix W and biases b
            # The activation function is a tanh.
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        # The last layer is linear, fully-connected
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    # u network takes x and t as input
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    # f network takes x and t as input
    def net_f(self, x, t):
        # Run u network
        u = self.net_u(x,t)
        # Calculate the gradient with respect to t
        u_t = tf.gradients(u, t)[0]
        # Calculate the gradient with respect to u
        u_x = tf.gradients(u, x)[0]
        # 2nd order derivative
        u_xx = tf.gradients(u_x, x)[0]
        # Evaluate f using the derivatives
        # This is Burgers' equation
        f = u_t + u*u_x - self.nu*u_xx
        
        return f
    
    def callback(self, loss):
        # This function will run at the end of every iteration
        if self.iteration%40 == 0:
            print("Step = {0:6d}, loss = {1:11.8f}".format(self.iteration,loss))
        self.iteration += 1
        
    def train(self):
        
        # Sets up the input for the NN
        # x_u: x locations of u data
        # t_u: t locations of u data
        # u: exact u observations for training
        # x_f: x locations for evaluating f residual
        # t_f: t locations for evaluating t residual
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        # Start optimization loop                                                                                                  
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
    
    def predict(self, X_star):
                
        # Simply run the NN using x and t data in X_star
        # and predict u and f at these locations.
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
               
        return u_star, f_star
    
if __name__ == "__main__": 
     
    # nu: coefficient in front of diffusion term
    nu = 0.01/np.pi
    noise = 0.0        

    # Number of u training points and f residual evaluation points
    N_u = 100
    N_f = 10000
    # Size of layers in NN
    # Input has dimension 2: (x,t)
    # Output has dimension 1: u(x,t) (scalar function)
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    # Load grid information and exact solution everywhere
    # (precomputed data)
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    # t and x discrete time and spatial steps
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    # Grid based on t and x discrete samples
    X, T = np.meshgrid(x,t)
    
    # Concatenate x and t locations
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)   

    # X has size: (size of t) x (size of x)
    # x: size 256
    # t: size 100
    # X: 100 x 256
        
    # Left side of boundary (t=0)  
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    # Lower side: x=-1
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    # Upper side: x=1
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    # Location of training points for u
    X_u_train = np.vstack([xx1, xx2, xx3])
    # Locations for f are chosen using a Latin Hypercube sampling
    # https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    # https://pythonhosted.org/pyDOE/randomized.html
    # 2:    number of dimensions
    # N_f:  number of samples for f_train
    # X_f_train has size: N_f x 2
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    # Locations of training points for f and u
    X_f_train = np.vstack((X_f_train, X_u_train))
    # Exact solution at X_u_train locations
    u_train = np.vstack([uu1, uu2, uu3])
    
    # Draw randomly without replacement N_u points
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # X_u_train:    location of training points for u
    # u_train:      exact solution at those points
    # X_f_train:    location of training points for f
    # By construction, f should be 0 at these points
        
    # Set up the model        
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    
    print("Starting training")

    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                

    print("Training of PILM completed successfully")
    print('Training time: %.2f [sec]' % (elapsed))

    print("PILM mode info:")
    print("Lower bound and upper bounds on x and t: ", lb, ub)
    print("Discretization points along x = ", len(x))
    print("Discretization points along t = ", len(t))
    print("Number of training data points for u (on the boundary) = ", N_u)
    print("Number of training locations for f (Latin hypercube) = ", N_f)
    print("Number of points in mesh (time dimension, spatial dimension): X/T = ", X.shape, T.shape)
    print("Number of points on boundary used for training = ", u_train.shape)
    print("Number of layers in NN for u(t,x) = ", len(layers))
    
    # Evaluate error using NN prediction

    u_pred, f_pred = model.predict(X_star)            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    print('Error u: %e' % (error_u))                     

    # Use to plot results
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')
    
    ####### Row 1 of figure: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 2 of figure: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)

    # Shows the plot in a figure
    # plt.show()

    # Save the figure to a file
    savefig('./figure_pilm_burgers')  
