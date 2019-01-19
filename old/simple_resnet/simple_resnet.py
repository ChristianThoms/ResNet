import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

total_layers = 25 #Specify how deep we want our network
units_between_stride = int(total_layers / 5)

def resUnit(input_layer,i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,64,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,64,[3,3],activation_fn=None)
        output = input_layer + part6
        return output

tf.reset_default_graph()
###############################
data = input_data.read_data_sets('data/fashion',one_hot=True)

label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)

train_Y = data.train.labels
test_Y = data.test.labels

training_iters = 200 
learning_rate = 0.001 
batch_size = 64

# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 10

#both placeholders are of type float
#x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])
###############################

x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32,name='input')
#label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
#label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(x,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = resUnit(layer1,j + (i*units_between_stride))
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output) + 1e-10, reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    #for i in range(training_iters):
    for i in range(40):
        print("i:", i)
        print("len", len(train_X)//batch_size, len(train_X), batch_size)
        #for batch in range(len(train_X)//batch_size):
        for batch in range(30):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(update, feed_dict={x: batch_x, y: batch_y})
            #loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        iter_loss = []
        iter_accuracy = []
        for batch in range(10):
            batch_x = test_X[batch*batch_size:min((batch+1)*batch_size,len(test_X))]
            batch_y = test_Y[batch*batch_size:min((batch+1)*batch_size,len(test_Y))]
            valid_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            iter_loss.append(valid_loss)
            iter_accuracy.append(test_acc)

        train_loss.append(loss)
        train_accuracy.append(acc)
        test_loss.append(sum(iter_loss)/10)
        test_accuracy.append(sum(iter_accuracy)/10)
        print("Testing Accuracy:","{:.5f}".format(test_accuracy[i]))
    summary_writer.close()
