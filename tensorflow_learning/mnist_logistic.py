import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

#定义模型权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

batch_xs, batch_ys = mnist.train.next_batch(batch_size)#数据loader

#模型结构
pred = tf.nn.softmax(tf.matmul(batch_xs, W) + b) # Softmax  x1*w1+x2*w2+……+b

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(batch_ys*tf.log(pred), reduction_indices=1))
#

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
total_batch = int(mnist.train.num_examples/batch_size)

# Start training
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord) 
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        
        # Loop over all batches
        for i in range(total_batch):
            
            # Fit training using batch data
            _, c = sess.run([optimizer, cost])#, feed_dict={x: batch_xs,
                                              #            y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(batch_ys, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
    coord.request_stop()  
    coord.join(threads)