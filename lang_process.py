import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

#mnist  = input_data.read_data_sets("/tmp/data/" , one_hot = True)
from create_feature_sets import create_features_and_labels
train_x,train_y,test_x,test_y = create_features_and_labels('/Users/avnish/LearningNewstuff/Data_Analysis/NLP/pos.txt','/Users/avnish/LearningNewstuff/Data_Analysis/NLP/neg.txt')

n_nodes_hiddenL1 = 500
n_nodes_hiddenL2 = 500
n_nodes_hiddenL3 = 500

n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None , len(train_x[0])])
y = tf.placeholder('float')

def neural_network(data):
	hidden_layer1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hiddenL1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hiddenL1]))}
	hidden_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenL1,n_nodes_hiddenL2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hiddenL2]))}
	hidden_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenL2,n_nodes_hiddenL3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hiddenL3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenL3,n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_layer2['weights']),hidden_layer2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_layer3['weights']),hidden_layer3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])

	return output

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

	
