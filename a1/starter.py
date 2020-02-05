
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import os

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):

    total_size = y.shape[0]
    y_predict = np.matmul(x,W) + b
    error = y_predict - y

    loss_function = np.sum(np.square(error))/ total_size
    decay_loss = (reg/2)* np.sum(np.square(W))

    mse = loss_function + decay_loss

    return mse

    # Your implementation here

def gradMSE(W, b, x, y, reg):
    total_size = y.shape[0]
    y_predict = np.matmul(x, W) + b
    error = y_predict - y

    W_grad = (np.matmul(np.transpose(x), error)/total_size) + reg * W
    b_grad =  np.sum(error) / total_size

    return W_grad, b_grad

    # Your implementation here


def calculate_accuracy(data, target):
    acc = np.sum((data > 0.5) == target) / data.shape[0]
    return acc

def grad_descent(W, b, x, y, val_data, val_target, test_data, test_target,  alpha, epochs, reg, error_tol, loss_type):
    # Your implementation here

    train_output = np.matmul(x, W) + b
    val_output = np.matmul(val_data, W) + b
    test_output = np.matmul(test_data, W) + b

    train_accuracy = [calculate_accuracy(train_output, y)]
    val_accuracy = [calculate_accuracy(val_output, val_target)]
    test_accuracy = [calculate_accuracy(test_output, test_target)]
    '''
        train_error = 0
        val_error = 0
        test_error = 0

        for i in range(y.shape[0]):
            if (train_output[i] > 0.5 and y[i] == 0) or(train_output[i] < 0.5 and y[i] == 1):
                train_error += 1

        for i in range(val_target.shape[0]):
            if (val_output[i] > 0.5 and val_target[i] == 0) or (val_output[i] < 0.5 and val_target[i] == 1):
                val_error += 1

        for i in range(test_target.shape[0]):
            if (test_output[i] > 0.5 and test_target[i] == 0) or(test_output[i] < 0.5 and test_target[i] == 1):
                test_error += 1

        train_accuracy = [ 1 - (train_error / train_output.shape[0])]
        val_accuracy = [ 1 - (val_error / val_output.shape[0])]
        test_accuracy = [ 1 - (test_error / test_output.shape[0])]
        '''

    train_loss = []
    val_loss = []
    test_loss = []

    if loss_type == 'MSE':
        train_loss = [MSE(W, b, x, y, reg)]
        val_loss = [MSE(W, b, val_data, val_target, reg)]
        test_loss = [MSE(W, b, test_data, test_target, reg)]

        for i in range(epochs):

            grad_W, grad_b = gradMSE(W, b, x, y, reg)
            W_update = W - alpha * grad_W
            b_update = b - alpha * grad_b

            train_loss.append(MSE(W_update, b_update, x, y, reg))
            val_loss.append(MSE(W_update, b_update, val_data, val_target, reg))
            test_loss.append(MSE(W_update, b_update, test_data, test_target, reg))

            train_output = np.matmul(x, W_update) + b_update
            val_output = np.matmul(val_data, W_update) + b_update
            test_output = np.matmul(test_data, W_update) + b_update

            train_accuracy.append(calculate_accuracy(train_output, y))
            val_accuracy.append(calculate_accuracy(val_output, val_target))
            test_accuracy.append(calculate_accuracy(test_output, test_target))

            diff_weight = np.linalg.norm(W_update - W)

            if diff_weight < error_tol:

                return W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy
            else:

                W = W_update
                b = b_update

        return W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy

    elif loss_type == 'CE':
        train_loss = [crossEntropyLoss(W, b, x, y, reg)]
        val_loss = [crossEntropyLoss(W, b, val_data, val_target, reg)]
        test_loss = [crossEntropyLoss(W, b, test_data, test_target, reg)]

        for i in range(epochs):

            grad_W, grad_b = gradCE(W, b, x, y, reg)
            W_update = W - alpha * grad_W
            b_update = b - alpha * grad_b

            train_loss.append(crossEntropyLoss(W_update, b_update, x, y, reg))
            val_loss.append(crossEntropyLoss(W_update, b_update, val_data, val_target, reg))
            test_loss.append(crossEntropyLoss(W_update, b_update, test_data, test_target, reg))

            train_output = np.matmul(x, W_update) + b_update
            val_output = np.matmul(val_data, W_update) + b_update
            test_output = np.matmul(test_data, W_update) + b_update

            train_accuracy.append(calculate_accuracy(train_output, y))
            val_accuracy.append(calculate_accuracy(val_output, val_target))
            test_accuracy.append(calculate_accuracy(test_output, test_target))

            diff_weight = np.linalg.norm(W_update - W)

            if diff_weight < error_tol:

                return W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy
            else:

                W = W_update
                b = b_update

        return W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy


    return W, b, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy

def normal_equation(x, y):

    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.transpose(x)), y)

    return W

def MSE_normal(W, x, y):

    total_size = y.shape[0]
    y_predict = np.matmul(x, W)
    error = y_predict - y

    mse = np.sum(np.square(error))/ total_size
    return mse


def plot_loss(iterations, train, val, test, l_rate, lamda, type):


    if type == 'Loss':
        plot_title = 'Loss Graph for ' + str(l_rate) + ' learning rate, ' + str(lamda) + ' lambda'
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Loss', 'Validation Loss', 'Test Loss'], loc='upper right')
        plt.show()
    elif type == 'Accuracy':
        plot_title = 'Accuracy Graph for ' + str(l_rate) + ' learning rate, '+ str(lamda) + ' lambda'
        plt.title(plot_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'], loc='lower right')
        plt.show()



def plot_part_three(iterations, train, val, test, l_rate, lamda, type):


    if type == 'Loss':
        plot_title = 'SGD Loss Graph for ' + str(l_rate) + ' learning rate, ' + str(lamda) + ' lambda'
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Loss', 'Validation Loss', 'Test Loss'], loc='upper right')
        plt.show()
    elif type == 'Accuracy':
        plot_title = 'SGD Accuracy Graph for ' + str(l_rate) + ' learning rate, '+ str(lamda) + ' lambda'
        plt.title(plot_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'], loc='lower right')
        plt.show()





def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    total_size = y.shape[0]
    x_sample = np.matmul(x, W) + b
    y_predict = 1 / ( 1 + np.exp((-1) * x_sample))

    CELoss = (np.sum((-1)*y*np.log(y_predict) - (1 - y) * np.log(1 - y_predict)) / total_size) + (reg/2) * np.sum(np.square(W))
    return CELoss


def gradCE(W, b, x, y, reg):
    # Your implementation here
    total_size = y.shape[0]
    x_sample = np.matmul(x, W) + b
    y_predict = 1 / (1 + np.exp((-1) * x_sample))
    error = y_predict - y

    W_grad = np.matmul(np.transpose(x), error)/total_size + reg * W
    b_grad = np.sum(error)/total_size

    return W_grad, b_grad


def buildGraph(loss="MSE", reg=1.0):
	#Initialize weight and bias tensors
    tf.compat.v1.set_random_seed(421)
    W = tf.Variable(tf.random.truncated_normal(shape=(784,1), stddev= 0.5), name="weight")
    bias = tf.Variable(0.0, name= "bias")
    train_data = tf.compat.v1.placeholder(tf.float32, [None, 784])
    train_target = tf.compat.v1.placeholder(tf.float32, [None, 1])


    if loss == "MSE":
        predict_target = tf.matmul(train_data, W) + bias
        loss = tf.compat.v1.losses.mean_squared_error(train_target, predict_target)
        reg_parameter = tf.nn.l2_loss(W)
        loss += reg * reg_parameter
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)


    elif loss == "CE":
        predict_target = tf.matmul(train_data, W) + bias
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(train_target, predict_target)
        reg_parameter = tf.nn.l2_loss(W)
        loss += reg * reg_parameter
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss)


    return W, bias, train_data, predict_target, train_target, loss, optimizer



    #part 1
'''
if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    trainData = np.reshape(trainData, (-1, 784))
    validData = np.reshape(validData, (-1, 784))
    testData = np.reshape(testData, (-1, 784))
    W = np.random.normal(0, 0.5, (trainData.shape[1], 1))
    b = 0

    alpha = 0.005
    lamda = 0
    epoch = 5000
    reg = 1 * 10 ** (-7)
    
    
    
    
    W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, alpha, epoch, lamda, reg, 'MSE')


    iterations = range(len(train_loss))

    plot_loss(iterations, train_loss, val_loss, test_loss, alpha, lamda,  'Loss')
    plot_loss(iterations, train_accuracy, val_accuracy, test_accuracy, alpha, lamda,  'Accuracy')

    
    print(train_accuracy[len(train_loss)-1])
    print(val_accuracy[len(train_loss) - 1])
    print(test_accuracy[len(train_loss) - 1])




'''



    '''
    when learning rate is 0.005
    train accuracy: 0.76
    valid accuracy: 0.67
    test accuracy: 0.75
    
    when learning rate is 0.001
    train accuracy: 0.65
    valid accuracy: 0.61
    test accuracy: 0.57
    
    when learning rate is 0.0001
    train accuracy: 0.5591
    valid accuracy: 0.5700
    test accuracy: 0.5517
    
    when learning rate is 0.005, lamda = 0.001
    train accuracy: 0.763
    valid accuracy: 0.679
    test accuracy: 0.752
    
    when learning rate is 0.005, lamda = 0.1
    train accuracy: 0.972
    valid accuracy: 0.98
    test accuracy: 0.965
    
    when learning rate is 0.005, lamda = 0.5
    train accuracy: 0.971
    valid accuracy: 0.97
    test accuracy: 0.9655
    '''

    '''
    part 1.5 code:
        W_least_square = normal_equation(trainData,trainTarget)
        MSE_norm = MSE_normal(W_least_square, trainData, trainTarget)
        
        starttime = os.times()[0]
        W_update, b_update, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = grad_descent(W, b, trainData, trainTarget, validData, validTarget, testData, testTarget, alpha, epoch, lamda, reg)
        mseGD = MSE(W_update, b_update, trainData, trainTarget, lamda)
        endTime = os.times()[0]
        
        GD_time = endTime - starttime


        out_train = np.matmul(trainData, W_update) + b_update
        out_valid = np.matmul(validData, W_update) + b_update
        out_test = np.matmul(testData, W_update) + b_update
    
    
        train_acc = calculate_accuracy(out_train,trainTarget)
        valid_acc = calculate_accuracy(out_valid, validTarget)
        test_acc = calculate_accuracy(out_test, testTarget)
    
    
        print(GD_time)
        print(mseGD)
        print(train_acc)
        print(valid_acc)
        print(test_acc)
        
    part 1.5 data:
        
        Normal Equation:
            compute time: 0.3100 s
            train accuracy: 0.984
            valid accuracy: 0.94
            test accuracy: 0.924
            MSE: 0.0232
            
            
            
        Batch GD:
            compute time: 23.89 s
            train accuracy: 0.76 
            valid accuracy: 0.67
            test accuracy: 0.75
            MSE: 0.528 
        
    '''

    #part 2


    #part 2.3 plot

    '''
    iterations = range(len(train_loss))

    plt.title('Logistic Regression vs Linear Regression')
    plt.ylabel('Loss')
    plt.xlabel('Number of iterations')
    plt.plot(iterations,train_loss)
    plt.plot(iterations,CEtrain_loss)
    plt.legend(['MSE Loss', 'CE Loss'], loc= 'upper right')
    plt.show()
    '''



    #end part 2.3



    '''
    iterations = range(len(train_loss))

    plot_loss(iterations, train_loss, val_loss, test_loss, alpha, lamda, 'Loss')
    plot_loss(iterations, train_accuracy, val_accuracy, test_accuracy, alpha, lamda, 'Accuracy')

    print(train_accuracy[len(train_loss) - 1])
    print(val_accuracy[len(train_loss) - 1])
    print(test_accuracy[len(train_loss) - 1])
    '''


    '''
    2.2: 
        
        performance for lambda = 0.1 alpha = 0.005 and 5000 epochs
        
        train accuracy: 0.974
        valid accuracy: 0.97
        test accuracy: 0.965
    '''

    #beginning of part 3


if __name__ == '__main__':
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    trainData = np.reshape(trainData, (-1, 784))
    validData = np.reshape(validData, (-1, 784))
    testData = np.reshape(testData, (-1, 784))

    W, bias, train_data, predict_target, train_target, loss, optimizer = buildGraph('MSE', reg=0.0)

    mini_batchsize = 500
    epoch = 700

    batch_number = int(trainData.shape[0]/mini_batchsize)

    train_loss = []
    valid_loss =[]
    test_loss = []

    train_accuracy = []
    valid_acuuracy =[]
    test_accuracy = []

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for i in range(epoch):
            for j in range(batch_number):

                train_batch = trainData[mini_batchsize*j:mini_batchsize*(j+1), :]
                test_batch = trainTarget[mini_batchsize*j:mini_batchsize*(j+1), :]
                _, t_loss = sess.run([optimizer, loss, predict_target], feed_dict={train_data: train_batch, train_target: test_batch})


            loss_train = sess.run(loss, feed_dict={train_data: trainData, train_target: trainTarget})
            loss_valid = sess.run(loss, feed_dict={train_data: validData, train_target: validTarget})
            loss_test = sess.run(loss, feed_dict={train_data: testData, train_target: testTarget})

            train_loss.append(loss_train)
            valid_loss.append(loss_valid)
            test_loss.append(loss_test)


            acc_train = calculate_accuracy(sess.run(predict_target, feed_dict={train_data: trainData, train_target: trainTarget}), trainTarget)
            acc_valid = calculate_accuracy(sess.run(predict_target, feed_dict={train_data: validData, train_target: validTarget}), validTarget)
            acc_test = calculate_accuracy(sess.run(predict_target, feed_dict={train_data: testData, train_target: testTarget}), testTarget)

            train_accuracy.append(acc_train)
            valid_acuuracy.append(acc_valid)
            test_accuracy.append(acc_test)

            #shuffule data
            np.random.seed(421 + i)
            np.random.shuffle(trainData)
            np.random.seed(421 + i)
            np.random.shuffle(trainTarget)



    iterations = range(len(train_loss))

    plot_part_three(iterations, train_loss, valid_loss, test_loss, 0.001, 0, 'Loss')
    plot_part_three(iterations, train_accuracy, valid_acuuracy, test_accuracy, 0.001, 0, 'Accuracy')





