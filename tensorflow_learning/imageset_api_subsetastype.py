#关于数据集
#子文件夹作为类
#参考https://blog.csdn.net/m0_37407756/article/details/80671961
#tf分为前端系统（负责构造计算图，就是op）还有后端系统（会话，负责计算图）
#数据读图分三种：预加载，placeholder&feed_dict(会有开销，只适合小型数据)，直接从文件读取(这种主要应对大型数据，前两种不适合大型数据)
#先建立名字队列，一次取出batch样本，然后再读样本

 # ROOT_FOLDER
       # |-------- SUBFOLDER (CLASS 0)
       # |             |
       # |             | ----- image1.jpg
       # |             | ----- image2.jpg
       # |             | ----- etc...
       # |             
       # |-------- SUBFOLDER (CLASS 1)
       # |             |
       # |             | ----- image1.jpg
       # |             | ----- image2.jpg
       # |             | ----- etc...
# 假设文件结构如上

from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
DATASET_PATH = r'./testdata' # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

# def reformat(samples,labels):
	# [图片数,高，宽，通道]


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
	imagepaths, labels = list(), list()
	dict={}
	if mode == 'file':
		# Read dataset file
		data = open(dataset_path, 'r').read().splitlines()
		for d in data:
			imagepaths.append(d.split(' ')[0])
			labels.append(int(d.split(' ')[1]))
	elif mode == 'folder':
		# An ID will be affected to each sub-folders by alphabetical order
		label = 0
		# List the directory
		
		try:  # Python 2
			classes = sorted(os.walk(dataset_path).next()[1])
			
		except Exception:  # Python 3
			classes = sorted(os.walk(dataset_path).__next__()[1])
		
		#得到类别，文件夹名字
		# List each sub-directory (the classes)
		for c in classes:
			c_dir = os.path.join(dataset_path, c)#拼成子文件夹全名
			try:  # Python 2
				walk = os.walk(c_dir).next()
			except Exception:  # Python 3
				walk = os.walk(c_dir).__next__()
			# Add each image to the training set
			for sample in walk[2]:#遍历这个子文件夹里面的图片
				# Only keeps jpeg images
				if sample.endswith('.png') or sample.endswith('.png'):
					imagepaths.append(os.path.join(c_dir, sample))#增加一个图的地址到list1
					labels.append(label)#把对应的label加入到第二个list2(这里可以自己拼路径和label)
			#拼完一个类到list之后
			dict.update({c:label})
			#print('拼出来的字典',dict)#{'trainingJunk_png': 0, 'trainingSymbols_png': 1}
			label += 1
	else:
		raise Exception("Unknown mode.")
	#前面是可以自己修改的部分，拼路径和拼label两个list
	#前面选取file指的是文件里找到地址列表
	
	# Convert to Tensor
	#将图片地址转为tensor型
	#imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
	#label也转成tensor
	#labels = tf.convert_to_tensor(labels, dtype=tf.int32)
	
	# Build a TF Queue, shuffle data
	
	#放入图地址（而不是整张图片，如果放入整张的话，就没有队列的意义了）
	#和label到队列，每次取其实是得到地址和label真值
	imagepaths2=imagepaths[:10]#给定部分下标，然后赋值给新的list变量
	labels2=labels[:10]
	print('在这里在这里:',len(labels2))
	image, label = tf.train.slice_input_producer([imagepaths2, labels2],#地址和label各自是listh,每次都从两条list里面取出前128个数据
												shuffle=True)
	
	#解码图地址
	# Read images from disk
	image = tf.read_file(image)
	image = tf.image.decode_jpeg(image, channels=CHANNELS)#image还是tensor类型
	#tf.decode_csv
	
	# Resize images to a common size
	image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
	image=tf.image.rgb_to_grayscale(image)   #(rgb转单通道的灰度)
	
	
	# Normalize
	image = image * 1.0/127.5 - 1.0
	
	
	# Create batches#batch产生器：data_loader
	#train.batch必须和producer配合使用
	X, Y = tf.train.batch([image, label], batch_size=batch_size,
						  capacity=batch_size * 8,
						  num_threads=4)

	return X, Y#建立好了X,Y的dataloader，run它会得到numpy型,run的时候不会再重新跑一边def函数

# Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 2
display_step = 100
dropout = 0.75





def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
		#reuse设置用于方便共享模型，训练的时候创建模型，测试的时候共享

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out

	
X, Y = read_images(DATASET_PATH, MODE, batch_size)#创建读图op,设置了128个batch
print(type(X),' | ',type(Y) )#其实还没与取出来，这只是一个loader
	
	
# tf.placeholder(tf.float32, shape=[None, 300, 300, 5])
labels = tf.placeholder(tf.float32, shape=[None, 10])

#创建模型op，这里直接用建立好的数据队列，直接输入到op，而没有通过placeholder&feed_dict来喂数据
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)
#换一个输入,reuse的话，就不是创建一个新的模型，是共享模型

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
	# Run the initializer
	sess.run(init)
	# Start the data queue
	tf.train.start_queue_runners()#充值队列,此时runX，Y就会得到numpy型，
	a=sess.run(X)
	# b=sess.run(X)#两次取到的图片，a，b不同，如果run了X之后，会得到前128个X和Y，如果再单独runY，是下一个batch的Y，
	#对于X,Y 可以取无限次，取完一个epoch之后，自动制作第二个epoch
	# plt.figure(1)
	# plt.imshow(a[0])
	# plt.figure(2)
	# plt.imshow(b[0])
	# plt.show()把取到的numpy图片显示出来
	print('获得图片:',a.shape)#128,64,64,3 ，里外都是numpy型
	
	# Training cycle
	for step in range(1, num_steps+1):
		if step % display_step == 0: 
			a,b=sess.run([X,Y])#同时run不要忘记[]
			print('X是：',type(a),'  |  ','Y是:',type(b))#图片/label数据在运算中是numpy类型
			# Run optimization and calculate batch loss and accuracy
			_, loss, acc = sess.run([train_op, loss_op, accuracy])
			print("Step " + str(step) + ", Minibatch Loss= " + \
				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
				  "{:.3f}".format(acc))
		else:
			# Only run the optimization op (backprop)
			sess.run(train_op)