import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):

    return np.maximum(x, 0)


def softmax(x):

    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def computeLayer(X, W, b):

    return np.matmul(X,W) + b

def CE(target, prediction):

    return -1 * np.sum(target * np.log(prediction)) / target.shape[0]



def gradCE(target, prediction):

    return prediction - target


def backprop_Wo(target, prediction, hidden_output):

    return np.matmul(np.transpose(hidden_output), (prediction-target))

def backprop_bo(target, prediction):
    ones = np.ones((1,target.shape[0]))

    return np.matmul(ones, (prediction-target))


def backprop_Wh(target, prediction, input_out, input, W_o):
    back_term = np.matmul((prediction - target), np.transpose(W_o))

    back_term = np.where(input_out < 0, 0, back_term)

    return np.matmul(np.transpose(input), back_term)




def backprop_bh(target, prediction, input_out, W_o):

    back_term = np.matmul((prediction - target),np.transpose(W_o))

    back_term = np.where(input_out < 0, 0, back_term)
    ones = np.ones((1, target.shape[0]))

    return np.matmul(ones, back_term)

def forward_prop(train_Data, W_h, b_h, W_o, b_o):

    S_h = computeLayer(train_Data, W_h, b_h)
    hidden_out = relu(S_h)
    output = softmax(computeLayer(hidden_out, W_o, b_o))
    return output, S_h, hidden_out



def learning(trainData, validData, testData, trainTarget, validTarget, testTarget, epochs, learning_rate, v_o, v_h, gamma, W_h, b_h, W_o, b_o):


    W_v_o = v_o
    W_v_h = v_h
    b_v_o = b_o
    b_v_h = b_h


    W_hidden = W_h
    b_hidden = b_h
    W_out = W_o
    b_out = b_o


    train_loss = []
    valid_loss = []
    test_loss = []

    train_acc = []
    valid_acc = []
    test_acc = []

    for i in range(epochs):
        print("Epoch: ", i)

        train_out, hidden_in, hidden_out = forward_prop(trainData, W_hidden, b_hidden, W_out, b_out)
        train_loss.append(CE(trainTarget, train_out))
        #calculate accuracy
        train_index = np.argmax(train_out, axis= 1)
        train_target_index = np.argmax(trainTarget, axis=1)
        compare = np.equal(train_index, train_target_index)
        train_acc.append(sum(compare == True)/ train_out.shape[0])


        valid_out,_,_ = forward_prop(validData, W_hidden, b_hidden, W_out, b_out)
        valid_loss.append(CE(validTarget, valid_out))
        # calculate accuracy
        valid_index = np.argmax(valid_out, axis=1)
        valid_target_index = np.argmax(validTarget, axis=1)
        compare = np.equal(valid_index, valid_target_index)
        valid_acc.append(sum(compare == True) / valid_out.shape[0])



        test_out,_,_ = forward_prop(testData, W_hidden, b_hidden, W_out, b_out)
        test_loss.append(CE(testTarget, test_out))
        #calculate accuracy
        test_index = np.argmax(test_out, axis=1)
        test_target_index = np.argmax(testTarget, axis=1)
        compare = np.equal(test_index, test_target_index)
        test_acc.append(sum(compare == True)/ test_out.shape[0])


        W_v_h = gamma * W_v_h + learning_rate * backprop_Wh(trainTarget, train_out, hidden_in, trainData, W_out)
        W_hidden = W_hidden - W_v_h
        b_v_h = gamma * b_v_h + learning_rate * backprop_bh(trainTarget, train_out, hidden_in, W_out)
        b_hidden = b_hidden - b_v_h


        W_v_o = gamma * W_v_o + learning_rate * backprop_Wo(trainTarget, train_out, hidden_out)
        W_out = W_out - W_v_o
        b_v_o = gamma * b_v_o + learning_rate * backprop_bo(trainTarget, train_out)
        b_out = b_out - b_v_o



    return train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc, W_out, b_out, W_hidden, b_hidden




def plot(iterations, train, val, test, type, title):


    if type == 'Loss':
        plot_title = title
        plt.title(plot_title)
        plt.ylabel('Loss')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Loss', 'Validation Loss', 'Test Loss'], loc='upper right')
        plt.show()
    elif type == 'Accuracy':
        plot_title = title
        plt.title(plot_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of iterations')
        plt.plot(iterations, train)
        plt.plot(iterations, val)
        plt.plot(iterations, test)
        plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'], loc='lower right')
        plt.show()


#part 2 functions


def cnn(input_data, keep_prob, weight, bias):
    # first convolution layer
    conv_out = tf.nn.conv2d(input_data, weight['w1'], strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, bias['b1'])

    # relu activation
    l1_out = tf.nn.relu(conv_out)

    # batch normalize
    mean, var = tf.nn.moments(l1_out, [0, 1, 2])
    ones_offset = tf.Variable(tf.ones([32]))
    zeros_offset = tf.Variable(tf.zeros([32]))
    normalized_batch = tf.nn.batch_normalization(l1_out, mean, var, zeros_offset, ones_offset, 1e-3)

    # maxpool layer
    max_pool_out = tf.nn.max_pool(normalized_batch, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # flatten
    reshaped_data = tf.reshape(max_pool_out, [-1, 6272])

    # first fully connected layer
    fc1_out = tf.add(tf.matmul(reshaped_data, weight['w2']), bias['b2'])

    # relu
    fc1_out = tf.nn.relu(fc1_out)

    # dropout
    fc1_out = tf.nn.dropout(fc1_out, keep_prob)

    # last fully connected layer with softmax out
    fc2_out = tf.add(tf.matmul(fc1_out, weight['w3']), bias['b3'])

    prediction = tf.nn.softmax(fc2_out)

    return prediction


def buildGraph(weight, bias, lam=0.0):
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, 10])
    prediction = cnn(x, keep_prob, weight, bias)

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    # add L2 regularization
    reg = lam * (tf.nn.l2_loss(weight['w2']) + tf.nn.l2_loss(weight['w3']))
    cost = tf.reduce_mean(entropy + reg)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return x, y, keep_prob, cost, optimizer, accuracy




'''

#Part 1 test

if __name__ == '__main__':

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()



    trainData = np.reshape(trainData, (-1, 784))
    validData = np.reshape(validData, (-1, 784))
    testData = np.reshape(testData, (-1, 784))


    new_train_target, new_val_target, new_test_target = convertOneHot(trainTarget, validTarget, testTarget)


    #initialization

    hidden_unit_size = 500
    gamma = 0.99
    learning_rate = 1e-5
    epochs = 200

    #input weight
    var_h = np.sqrt(2/(trainData.shape[1] + hidden_unit_size))
    W_hidden = np.random.normal(0,var_h,(trainData.shape[1], hidden_unit_size))
    b_hidden = np.zeros((1, hidden_unit_size))
    v_h = np.full((trainData.shape[1], hidden_unit_size), 1e-5)

    #output weight
    var_o = np.sqrt(2 / (hidden_unit_size+10))
    W_out = np.random.normal(0, var_o, (hidden_unit_size, 10))
    b_out = np.zeros((1, 10))
    v_o = np.full((hidden_unit_size,10), 1e-5)

    train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc, W_out, b_out, W_hidden, b_hidden = learning(trainData, validData, testData, new_train_target, new_val_target, new_test_target, epochs, learning_rate, v_o, v_h, gamma, W_hidden, b_hidden, W_out, b_out)


    iterations = range(len(train_loss))

    plot(iterations, train_loss, valid_loss, test_loss, type= 'Loss', title= 'Loss Graph Hidden unit size = 500')
    plot(iterations, train_acc, valid_acc, test_acc, type= 'Accuracy', title= 'Default Accuracy Graph Hidden unit size = 500')


    size = len(train_loss) -1
    print("The final Training Loss: ", train_loss[size])
    print("The final Validation Loss: ", valid_loss[size])
    print("The final Test Loss: ", test_loss[size])


    print("The final Training Accuracy: ", train_acc[size])
    print("The final Validation Accuracy: ", valid_acc[size])
    print("The final Test Accuracy: ", test_acc[size])

'''

#part 2 test

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()


#weights
weight={
    'w1': tf.get_variable(name='W1',shape=(3,3,1,32),initializer= tf.contrib.layers.xavier_initializer()),
    'w2': tf.get_variable(name='W2',shape=(6272, 784),initializer= tf.contrib.layers.xavier_initializer()),
    'w3': tf.get_variable(name='W3',shape=(784,10), initializer= tf.contrib.layers.xavier_initializer())
}
#bias
bias={
    'b1': tf.get_variable(name='B1',shape=(32), initializer= tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable(name='B2',shape=(784), initializer= tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable(name='B3',shape=(10), initializer= tf.contrib.layers.xavier_initializer())
}

trainData = np.reshape(trainData,(-1,28,28,1))
validData = np.reshape(validData, (-1,28,28,1))
testData = np.reshape(testData, (-1,28,28,1))

train_loss = []
valid_loss = []
test_loss = []
train_acc = []
valid_acc = []
test_acc = []

iterations = 50
batch_size = 32
batch_number = int(len(trainData) / 32)
prob = 0.5
lam = 0.0
x, y, keep_prob, cost, optimizer, accuracy = buildGraph(weight, bias, lam=lam)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        trainData, trainTarget = shuffle(trainData, trainTarget)
        new_train_target, new_val_target, new_test_target = convertOneHot(trainTarget, validTarget, testTarget)
        for j in range(batch_number):
            batch_trainData = trainData[j * batch_size:min((j * batch_size + batch_size), len(trainData))]
            batch_trainTarget = new_train_target[
                                j * batch_size:min((j * batch_size + batch_size), len(new_train_target))]

            opt = sess.run([optimizer], feed_dict={x: batch_trainData, y: batch_trainTarget, keep_prob: prob})

        trainLoss, trainAccuracy = sess.run([cost, accuracy],
                                            feed_dict={x: trainData, y: new_train_target, keep_prob: prob})
        validLoss, validAccuracy = sess.run([cost, accuracy],
                                            feed_dict={x: validData, y: new_val_target, keep_prob: prob})
        testLoss, testAccuracy = sess.run([cost, accuracy],
                                          feed_dict={x: testData, y: new_test_target, keep_prob: prob})

        train_loss.append(trainLoss)
        valid_loss.append(validLoss)
        test_loss.append(testLoss)

        train_acc.append(trainAccuracy)
        valid_acc.append(validAccuracy)
        test_acc.append(testAccuracy)
        print(i)



steps = range(len(train_loss))

plot(steps, train_loss, valid_loss, test_loss, type='Loss', title='Loss Graph for Dropout = 0.5')
plot(steps, train_acc, valid_acc, test_acc, type='Accuracy', title='Accuracy Graph for Dropout = 0.5')

size = len(train_loss) - 1

print("The final Training Loss: ", train_loss[size])
print("The final Validation Loss: ", valid_loss[size])
print("The final Test Loss: ", test_loss[size])

print("The final Training Accuracy: ", train_acc[size])
print("The final Validation Accuracy: ", valid_acc[size])
print("The final Test Accuracy: ", test_acc[size])



#part 1 data
'''
#default setting
The final Training Loss:  0.0014324476152077397
The final Validation Loss:  0.06258625582924553
The final Test Loss:  0.06505281651356519
The final Training Accuracy:  0.9964
The final Validation Accuracy:  0.9091666666666667
The final Test Accuracy:  0.907856093979442


#hidden unit 100
The final Training Loss:  0.0155735578351207
The final Validation Loss:  0.05109989429835016
The final Test Loss:  0.049849414125831874
The final Training Accuracy:  0.9513
The final Validation Accuracy:  0.8901666666666667
The final Test Accuracy:  0.893906020558003

#hidden unit 500 

The final Training Loss:  0.04930619540430964
The final Validation Loss:  0.5593368411805492
The final Test Loss:  0.6003995111823541
The final Training Accuracy:  0.9849
The final Validation Accuracy:  0.9036666666666666
The final Test Accuracy:  0.9056534508076358

#hidden unit 2000

The final Training Loss:  0.00046847270620087257
The final Validation Loss:  0.07239799515201899
The final Test Loss:  0.07692054471442997
The final Training Accuracy:  0.9989
The final Validation Accuracy:  0.914
The final Test Accuracy:  0.9140969162995595

'''

#part 2 data
'''
#default model
The final Training Loss:  1.481191
The final Validation Loss:  1.5316507
The final Test Loss:  1.540617
The final Training Accuracy:  0.98
The final Validation Accuracy:  0.9295
The final Test Accuracy:  0.92033774



#2.3 L2 reg investigation

lambda = 0.01 

The final Training Loss:  1.5530559
The final Validation Loss:  1.5954375
The final Test Loss:  1.5941565
The final Training Accuracy:  0.9702
The final Validation Accuracy:  0.92516667
The final Test Accuracy:  0.9276799

lambda = 0.1 

The final Training Loss:  1.6887094
The final Validation Loss:  1.7003801
The final Test Loss:  1.6961371
The final Training Accuracy:  0.9173
The final Validation Accuracy:  0.90566665
The final Test Accuracy:  0.9085903

lambda = 0.5

The final Training Loss:  1.9303219
The final Validation Loss:  1.9369711
The final Test Loss:  1.9296163
The final Training Accuracy:  0.8803
The final Validation Accuracy:  0.87166667
The final Test Accuracy:  0.8784875



#2.3 Dropout

Dropout = 0.9

The final Training Loss:  1.4816929
The final Validation Loss:  1.5347552
The final Test Loss:  1.535001
The final Training Accuracy:  0.9796
The final Validation Accuracy:  0.9266667
The final Test Accuracy:  0.9258444

Dropout = 0.75

The final Training Loss:  1.478655
The final Validation Loss:  1.5358737
The final Test Loss:  1.5381374
The final Training Accuracy:  0.9829
The final Validation Accuracy:  0.92466664
The final Test Accuracy:  0.9240088

Dropout = 0.5

The final Training Loss:  1.4805317
The final Validation Loss:  1.5313472
The final Test Loss:  1.5335038
The final Training Accuracy:  0.9809
The final Validation Accuracy:  0.93
The final Test Accuracy:  0.928047


'''









