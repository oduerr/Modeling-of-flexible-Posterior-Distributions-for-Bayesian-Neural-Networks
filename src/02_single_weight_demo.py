import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
print('Tensorflow version: ', tf.__version__)
import tensorflow_probability as tfp
#print(tf.config.experimental.list_physical_devices('GPU'))
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('TFP version: ',tfp.__version__)
import scipy.stats as stats
import seaborn as sns
import tqdm
import sys
import ctypes
from IPython.display import display, clear_output, HTML
from numpy import trapz
#sys.path.append('../')
from src.vimlts_minimal import VIMLTS
print(VIMLTS)
tfd=tfp.distributions
show_plots = False
epsilon=tf.constant(0.001)

def kl_divergence(q, p):
    """
        Calculates the kl divergence.
    """
    # Avoid numerical problems
    mask = ((q > 1e-08) | (p > 1e-08)) & (np.isinf(q)==0) & (np.isinf(p)==0)
    return tf.math.reduce_sum(tf.where(mask, tf.math.log(q/p), 0))

import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
tfd=tfp.distributions
log=tf.math.log

####
y = np.array([ -4,  -3, -2, 1.5, 3, 3.4 ], dtype=np.float32)
y = np.array([ -4,  -3.5, -2, 2.5, 3, 3.4 ], dtype=np.float32)
# Learning rate
lr_start=0.008
lr_end=0.003
epochs = 50
sigma = 1
num_samples = 10

def get_lr(current_epoch):
    """
        Returns the learning rate depending to the current epoch.
    """
    lr_dyn=(epochs-current_epoch)/epochs
    return lr_start*lr_dyn+lr_end*(1-lr_dyn)

# Variational parameters init
a_init=(1.0,)
b_init=(0.,)
alpha_init = (1,)
beta_init = (0,)

#ans=6#Mbox('Initialization', 'Using a high slope for the Bernstein polynomial initialization?', 4)
#if(ans==6):
    # Using a high slope will help to find a mode that creates a bimodal fit
delta_theta_high_slope_init = (-6,2.,-6.,-6.,-6.,8.,-6.,-6.,-6,3.)
delta_theta_init=delta_theta_high_slope_init
#else:
#    delta_theta_gaussian_like_init = (-5,1.,0.8,0.8,0.6,0.4,0.2,0.2,1.,1.5)
#    delta_theta_init=delta_theta_gaussian_like_init

lambda_init=a_init+b_init+delta_theta_init+alpha_init+beta_init
num_lambda=np.shape(lambda_init)[0]
lambda_tunable = tf.Variable(lambda_init,dtype='float32')

# Instance of a variational distribution as VIMLTS
q_dist=VIMLTS(np.shape(delta_theta_init)[0])

prior_dist=tfd.Normal(loc=0.,scale=1.)
ytensor = tf.Variable(y.reshape([len(y),1]))


q_dist.update_lambda_param(lambda_tunable)
### Plotting the Inital Distribution
if (True):
    pp,ww = q_dist.get_target_dist()
    plt.plot(ww, pp)
    plt.legend()
    plt.grid()
    plt.title(r'$q_{init}$')
    plt.ylabel(r'$q(w)$')
    plt.xlabel(r'$w$');
    plt.show()
#%%
##Testing the sampling of the distribution
if (False):
    zs = np.ndarray((1000), dtype='float32')
    ws = np.ndarray((1000), dtype='float32')
    for i in range(1000):
        z, w = q_dist.get_sample_w()
        ws[i] = w.numpy()
        zs[i] = z.numpy()

    plt.hist(ws,100)
    plt.title('Samples from the untrained variational distribution')
    plt.show()

#%%


#%%
print('Hallo Gallo')


import time
start = time.time()

#@tf.function
def train():
    global i, lambda_tunable
    losses_kl = np.ndarray(epochs)
    losses_nll = np.ndarray(epochs)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch([lambda_tunable])

            # Parameter update --------------------------------------------------
            q_dist.update_lambda_param(lambda_tunable)

            # iterarte over psi (number of samples)
            for current_sample in range(num_samples):
                # l_nll ------------------------------------------
                z_sample, w_sample = q_dist.get_sample_w()

                # Likelihood
                y_prob = tfd.Cauchy(loc=w_sample, scale=sigma)
                # loss_nll
                l_nll_sample = -tf.reduce_sum(y_prob.log_prob(ytensor))
                l_nll_sample = tf.reshape(l_nll_sample, [-1])
                if current_sample == 0:
                    l_nll_list = l_nll_sample
                else:
                    l_nll_list = tf.concat([l_nll_list, tf.reshape(l_nll_sample, [-1])], axis=0)

                # l_kl ------------------------------------------
                variational_dist_sample = q_dist.get_target_dist_for_z_eps(z_sample, epsilon)
                prior_sample = prior_dist.prob(w_sample)
                l_kl_sample = kl_divergence(variational_dist_sample, prior_sample)
                l_kl_sample = tf.reshape(l_kl_sample, [-1])
                if current_sample == 0:
                    l_kl_list = l_kl_sample
                else:
                    l_kl_list = tf.concat([l_kl_list, l_kl_sample], axis=0)

            l_nll = tf.reduce_mean(l_nll_list, axis=0)
            l_kl = tf.reduce_mean(l_kl_list, axis=0)

            losses_kl[i] = l_kl.numpy()
            losses_nll[i] = l_nll.numpy()
            # loss --------------------------------------
            loss = l_nll + l_kl
        grads = tape.gradient(loss, lambda_tunable)


        # Check for NaNs in gradient
        if True in tf.math.is_nan(grads):
            print("+++++++++NaN in gradient+++++++++++++")
            print("Parameter:", w_sample.numpy())
            print("Grads:", grads)
            continue

        # Update the variational parameter
        lr = get_lr(i)
        lambda_tunable = tf.Variable(lambda_tunable - lr * grads)

        # Prints and history
        if i % 10 == 0 or i < 10 or i == epochs - 1:
            print("i", i, "\tloss \t", loss.numpy(), "\tl_nll\t", l_nll.numpy(), "\tl_kl\t", l_kl.numpy(), "\tlr ", lr, "\r ", lambda_tunable)

    return(losses_nll, losses_kl)


losses_nll, losses_kl = train()
lambda_tunable
print('Epochs ', epochs, 'Time ', time.time() - start)

#### Plotting the resulting weight
plt.plot(losses_nll + losses_kl,label=r"total")
plt.plot(losses_nll,label=r"NLL")
plt.plot(losses_kl,label=r"KL")
plt.legend()
plt.title('Losses')
plt.show()

#### Plotting the resulting weight
pp,ww = q_dist.get_target_dist()
plt.plot(ww,pp)
plt.legend()
plt.grid()
plt.title('After fitting')
plt.ylabel(r'$q(w)$')
plt.xlabel(r'$w$');
plt.show()

#### Plotting the resulting
sample_list=[]
for i in range(500):
    z_sample,w_sample=q_dist.get_sample_w()
    sample_list.append(w_sample.numpy()[0])
ww=np.linspace(-12.,12.,num=int(1e3))
w_VIMLTS_density_kde = stats.gaussian_kde(sample_list);
plt.plot(ww,w_VIMLTS_density_kde(ww),label=r"VIMLTS - $q_{w}$");
plt.title('After fitting Sampling')
plt.show();



