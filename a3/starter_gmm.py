import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import collections
import sys

data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)



# Distance function for GMM
def distanceFunc(X, MU):
    X = tf.expand_dims(X, 0)
    MU = tf.expand_dims(MU, 1)

    temp = tf.reduce_sum(tf.square(tf.subtract(X,MU)), 2)
    # Outputs
    pair_dist = tf.transpose(temp)
    # pair_dist: is the squared pairwise distance matrix (NxK)


    return pair_dist
    

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    '''
    xmu = distanceFunc(X,mu)
    exp = -1*tf.div(xmu,tf.transpose(2*sigma))
    coeff = -1*tf.log((2*np.pi)**(dim/2)*sigma)
    pdf = tf.transpose(coeff) + exp
    '''
   
    dim = tf.to_float(tf.rank(X))
    sigma = tf.squeeze(sigma)
    coef = tf.log(2 * np.pi * sigma)
    exp = tf.divide(distanceFunc(X,mu), (2 * sigma))
    mul = -0.5*dim * coef
    PDF = mul - exp
    return PDF
    

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    log_pi = tf.squeeze(log_pi)
    sum_log = hlp.reduce_logsumexp(tf.add(log_PDF, log_pi), keep_dims= True)

    log_post = tf.subtract(tf.add(log_PDF, log_pi), sum_log)

    return log_post

def compute_loss(X, mu, sigma, pi):
    log_pi = tf.squeeze(hlp.logsoftmax(pi))
    log_pdf = log_GaussPDF(X, mu, sigma)

    log_loss = -1*tf.reduce_sum(hlp.reduce_logsumexp(log_pdf + log_pi, keep_dims= True))
    return log_loss, log_pdf, log_pi



def buildGraph(K, D, lr, stdv):
    X = tf.placeholder("float", [None, D])
    mu = tf.Variable(tf.random_normal([K, D], stddev = stdv))
    sigma = tf.exp(tf.Variable(tf.random_normal([K, 1], stddev = stdv)))
    pi = tf.Variable(tf.random_normal([K, 1], stddev = stdv))

    loss, log_pdf, log_pi = compute_loss(X, mu, sigma, pi)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.99).minimize(loss)
    
    pred = tf.argmax(tf.nn.softmax(log_posterior(log_pdf, log_pi)), 1)

    return X, mu, sigma, pi, loss, optimizer, pred




def Mog(K, data, use_valid= False):
    
    
    [num_pts, dim] = np.shape(data)
    is_valid = use_valid
    # For Validation set
    if is_valid:
      valid_batch = int(num_pts / 3.0)
      np.random.seed(45689)
      rnd_idx = np.arange(num_pts)
      np.random.shuffle(rnd_idx)
      val_data = data[rnd_idx[:valid_batch]]
      data = data[rnd_idx[valid_batch:]]



    train_loss_list = []
    valid_loss_list = []
    learning_rate = 0.1
    stddev = 0.05
    X, mu, sigma, pi, loss, optimizer, pred = buildGraph(K, dim, learning_rate, stddev)

    epochs = 300

    centers = np.zeros(shape=[K,dim])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            center, train_loss, predict, _ = sess.run([mu, loss, pred, optimizer], feed_dict={X:data})
            train_loss_list.append(train_loss)
            centers = center
            if use_valid:
                val_loss = sess.run([loss], feed_dict={X:val_data})
                valid_loss_list.append(val_loss)

        predict =np.int32(predict)

    return train_loss_list, valid_loss_list, centers, predict, data

def plot_loss(iterations, train, title, loss_name):
        loss = loss_name
        plot_title = title
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of Epochs')
        plt.plot(iterations, train)

        plt.legend([loss], loc='best')
        plt.show()

def plot_cluster(cluster, centers, data, title):
    plot_title = title
    plt.title(plot_title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(data[:,0], data[:,1], s=25, c=cluster, cmap=plt.get_cmap('Set3'), alpha=0.5)
    plt.scatter(centers[:,0], centers[:,1], s=50, c="black", cmap=plt.get_cmap('Set1'), marker='*', linewidths=1)
    plt.grid()
    plt.show()

def plot_multi_loss(iterations, l1,l2,l3,l4,l5, title):
        plot_title = title
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of Epochs')
        plt.plot(iterations, l1)
        plt.plot(iterations, l2)
        plt.plot(iterations, l3)
        plt.plot(iterations, l4)
        plt.plot(iterations, l5)


        plt.legend(['K=5', 'K=10', 'K=15','K=20','K=30'], loc='best')
        plt.show()



train_loss_list, valid_loss_list, centers, pred, final_data = Mog(5, data, use_valid=True)
iterations = range(len(valid_loss_list))
plot_loss(iterations, valid_loss_list, 'Validation Loss K=5', 'Validation Loss')
plot_cluster(pred,centers,final_data, 'Cluster K=5')


#Section 2.2.1
'''
Training loss: 17132.316
'''

#Section 2.2.2
'''
K = 1:
Validation Loss = 11651.518

K = 2:
Validation Loss = 7987.4756

K = 3:
Validation Loss = 5630.154

K = 4:
Validation Loss = 5630.721

K = 5:
Validation Loss = 5630.515
'''

#Section 2.2.3
data2 = np.load('data100D.npy')
_, v_loss5, _, _, _ = Mog(5, data2,use_valid=True)
_, v_loss10, _, _, _ = Mog(10, data2,use_valid=True)
_, v_loss15, _, _, _ = Mog(15, data2,use_valid=True)
_, v_loss20, _, _, _ = Mog(20, data2,use_valid=True)
_, v_loss30, _, _, _ = Mog(30, data2,use_valid=True)

iterations = range(len(v_loss5))
plot_multi_loss(iterations, v_loss5, v_loss10, v_loss15, v_loss20, v_loss30, 'Validation Loss on data100D')

'''
k = 5 validation loss: 21952.305

k = 10 validation loss: 22082.703

k = 15 validation loss: 21447.246

k = 20 validation loss: 21510.793

k = 30 validation loss: 21421.03
'''