 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


#集成函数
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
		padding='SAME', groups=1):
	"""Create a convolution layer.
	Adapted from: https://github.com/ethereon/caffe-tensorflow
	"""
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])
    # Create lambda function for the convolution
	convolve = lambda i, k: tf.nn.conv2d(i, k,strides=[1, stride_y, stride_x, 1],padding=padding)
	with tf.variable_scope(name) as scope:
		# Create tf variables for the weights and biases of the conv layer
		weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,num_filters],dtype=tf.float32)
		#tf.get_variable共享变量											
		#weights.name=name.weights
		#一、name_scope
		#(1)姓:scope,名:variable的name  名第一种情况：变量的名都一样，那就会显示姓作为前缀
		#https://blog.csdn.net/qq_27825451/article/details/82349984
		#（2）若大家的名都不同，那就直接显示名，而不显示姓作为前缀
		#(3)name_scope不允许共享变量，即一个域之内不能重'name
		#(如果姓重名了，则姓加上_1 _2后缀)
		#二、vari_scope
		#(4)varialbe_scope允许共享变量scope.reuse_variables()  # 设置共享变量，然后再创建重名的
		# 即	variable_scope/var1:0
			# variable_scope/var1:0
		#如果不设置reuse（只能对之后的一次创建有效），又重名的话，则在名加_1_2后缀
		#重姓：with tf.variable_scope(foo_scope, reuse=True):  #直接指定前面的那个variable_scope
		
													
		biases = tf.get_variable('biases', shape=[num_filters])
		#biases.name=name.biases
		
	if groups == 1:#非群卷积
		conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
	else:
        # Split input and weights and convolve them separately
		input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
		weight_groups = tf.split(axis=3, num_or_size_splits=groups,value=weights)
		output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
		conv = tf.concat(axis=3, values=output_groups)

    # Add biases
	bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    # Apply relu function
	relu = tf.nn.relu(bias, name=scope.name)
	return relu

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)
						  
def fc(x, num_in, num_out, name, relu=True):
	"""Create a fully connected layer."""
	with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
		weights = tf.get_variable('weights', shape=[num_in, num_out],trainable=True)
		biases = tf.get_variable('biases', [num_out], trainable=True)
		# Matrix multiply weights and inputs and add bias
		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

	if relu:
		# Apply ReLu non linearity
		relu = tf.nn.relu(act)
		return relu
	else:
		return act

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)
def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)	

#原生函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#tf.truncated_normal(shape, mean, stddev)维度，mean是均值，stddev是标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
	
	
	
	
	
	
	
	
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

                                        # output size 14x14x32

	
		
## conv1 layer ##
# W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
# h_pool1 = max_pool_2x2(h_conv1) 
		 
#conv1 layer
conv1=conv(x_image, 5, 5, 32, 1, 1, 'conv1',padding='SAME', groups=1)
h_pool1 = max_pool(conv1, 2, 2, 2, 2, padding='SAME', name='max1')
print('第一层输出:',h_pool1)#14*14*32
# ## conv2 layer ##
# W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
# h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

conv2=conv(h_pool1, 5, 5, 64, 1, 1, 'conv2',padding='SAME', groups=1)
h_pool2 = max_pool(conv2, 2, 2, 2, 2, padding='SAME', name='max1')
print('2conv的输出:',h_pool2.shape)

# ## func1 layer ##
# W_fc1 = weight_variable([7*7*64, 1024])
# b_fc1 = bias_variable([1024])
# # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1=fc(h_pool2_flat,7*7*64, 1024, 'fc1', relu=True)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

train_line=[]
train_line=[]

step_line=[]

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs: batch_xs,ys: batch_ys, keep_prob: 0.5})
	if i % 50 == 0:
		line_acc=compute_accuracy(mnist.test.images, mnist.test.labels)
		print(line_acc)
		
	
		
		
