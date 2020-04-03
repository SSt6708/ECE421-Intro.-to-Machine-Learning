import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import collections

# Loading data
data = np.load('/content/drive/My Drive/ECE421/Labs/a3/data2D.npy')
#data = np.load('data100D.npy')


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)

    X = tf.expand_dims(X, 0)
    MU = tf.expand_dims(MU, 1)

    temp = tf.reduce_sum(tf.square(tf.subtract(X,MU)), 2)
    # Outputs
    pair_dist = tf.transpose(temp)
    # pair_dist: is the squared pairwise distance matrix (NxK)


    return pair_dist

def buildGraph(K, D):
    X = tf.placeholder("float", shape=[None, D])
    MU = tf.Variable(tf.truncated_normal([K,D], stddev=0.05))
    dist = distanceFunc(X,MU)
    loss = tf.reduce_sum(tf.reduce_min(dist, axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    
    return X, MU, loss, optimizer

def k_means(K, data, use_valid= False):
    is_valid = use_valid
    [num_pts, dim] = np.shape(data)
    # For Validation set
    if is_valid:
      valid_batch = int(num_pts / 3.0)
      np.random.seed(45689)
      rnd_idx = np.arange(num_pts)
      np.random.shuffle(rnd_idx)
      val_data = data[rnd_idx[:valid_batch]]
      data = data[rnd_idx[valid_batch:]]

    
    X, MU, loss, optimizer = buildGraph(K,dim)

    epochs = 500

    k_means_loss = []
    k_means_val_loss = []

    centers = np.zeros(shape=[K,dim])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            
            center, train_loss, opt = sess.run([MU, loss, optimizer], feed_dict={X: data})
            centers = center
            k_means_loss.append(train_loss)
            if use_valid:
                val_loss = sess.run([loss], feed_dict={X:val_data})
                k_means_val_loss.append(val_loss)


        c = tf.argmin(distanceFunc(X,MU), axis=1)
        cluster = sess.run([c], feed_dict={X:data, MU: center})


    return k_means_loss, k_means_val_loss, cluster, centers



def plot_loss(iterations, train, title, loss_name):
        loss = loss_name
        plot_title = title
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of clusters')
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



def countPercent(cluster, data):
    size = data.shape[0]
    table = collections.Counter(cluster[0])

    for keys in table:
        percent = (table[keys]/size) * 100
        table[keys] = round(percent, 2)
    return table




#Section 1.1
k_means_loss,k_means_val_loss, cluster, centers = k_means(3, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=3')

#Section 1.2

'''
#K = 1
k_means_loss,k_means_val_loss, cluster, centers = k_means(1, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss K=1', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=1')
table = countPercent(cluster,data)
print(table)


#K = 2
k_means_loss,k_means_val_loss, cluster, centers = k_means(2, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss K=2', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=2')
table = countPercent(cluster,data)
print(table)

#K = 3
k_means_loss,k_means_val_loss, cluster, centers = k_means(3, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss K=3', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=3')
table = countPercent(cluster,data)
print(table)

#K = 4
k_means_loss,k_means_val_loss, cluster, centers = k_means(4, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss K=4', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=4')
table = countPercent(cluster,data)
print(table)
'''

#K = 5
k_means_loss,k_means_val_loss, cluster, centers = k_means(5, data)
iterations = range(len(k_means_loss))
plot_loss(iterations, k_means_loss, 'K_means Loss K=5', 'training_loss')
print(k_means_loss[-1])
plot_cluster(cluster, centers, data, 'K-Means Clustering K=5')
table = countPercent(cluster,data)
print(table)


#Section 1.3
k_means_loss1,k_means_val_loss1, cluster1, centers1 = k_means(1, data, use_valid=True)
k_means_loss2,k_means_val_loss2, cluster2, centers2 = k_means(2, data, use_valid=True)
k_means_loss3,k_means_val_loss3, cluster3, centers3 = k_means(3, data, use_valid=True)
k_means_loss4,k_means_val_loss4, cluster4, centers4 = k_means(4, data, use_valid=True)
k_means_loss5,k_means_val_loss5, cluster5, centers5 = k_means(5, data, use_valid=True)

val_loss_1 = k_means_val_loss1[-1]
val_loss_2 = k_means_val_loss2[-1]
val_loss_3 = k_means_val_loss3[-1]
val_loss_4 = k_means_val_loss4[-1]
val_loss_5 = k_means_val_loss5[-1]
print("Final Validation Loss for K = 1: ",val_loss_1)
print("Final Validation Loss for K = 2: ",val_loss_2)
print("Final Validation Loss for K = 3: ",val_loss_3)
print("Final Validation Loss for K = 4: ",val_loss_4)
print("Final Validation Loss for K = 5: ",val_loss_5)

val_iteration = [1,2,3,4,5]
val_loss = [val_loss_1,val_loss_2,val_loss_3,val_loss_4,val_loss_5]
plot_loss(val_iteration, val_loss, "Validation Loss", "Validation Loss")




#Data
'''
Section 1.1: 

k = 3, final loss = 5110.946



Section 1.2:

k = 1, final loss = 38453.492
Percentage of each class K = 1: {0: 100.0}

k = 2, final loss = 9203.354
Percentage of each class K = 2: {0: 50.45, 1: 49.55}

k = 3, final loss = 5110.946
Percentage of each class K = 3: {0: 38.13, 1: 38.06, 2: 23.81}

k = 4, final loss = 3374.0386
Percentage of each class K = 4: {2: 37.3, 3: 37.13, 0: 13.53, 1: 12.04}

k = 5, final loss = 2847.7358
Percentage of each class K = 5: {1: 36.24, 4: 35.75, 2: 10.67, 0: 8.84, 3: 8.5}



Section 1.3:

K = 1, validation Loss = 12873.747

K = 2, validation Loss = 2960.672

K = 3, validation Loss = 1629.3097

K = 4, validation Loss = 1054.6196

K = 5, validation Loss = 900.92377

'''