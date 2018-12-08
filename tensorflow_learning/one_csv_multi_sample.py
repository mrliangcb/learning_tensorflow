#一个csv文件，多个样本
#https://blog.csdn.net/m0_37407756/article/details/80671961
import tensorflow as tf  
filenames = [r'C:\lcb\learning_python_git\learning_tensorflow\tensorflow_learning\test.csv'] 

def get_data():
	## filenames = tf.train.match_filenames_once('.\data\*.csv') 
	filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  #名字放入文件名队列，一个epoch，打乱，
																	#如果名字队列要绑定多几个值就[image_name,value]
	# [B]
	# [A]
	# [C]
	#[结束]
	#定义一个reader，还没开始读
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])#分别读两列，有多少列就要读多少，等号前有多少变量
	#拿到的是tensor类型，可以串起来
	#features = tf.pack([example,label])

	#tf.decode_csv(
		# records,
		# record_defaults,#如果为空
		# field_delim=None,#分隔符默认为，
		# name=None
		# )




	# 使用tf.train.batch()会多加了一个样本队列和一个QueueRunner。Decoder解后数据会进入这个队列，再批量出队。  
	# 虽然这里只有一个Reader，但可以设置多线程，相应增加线程数会提高读取速度，但并不是线程越多越好。  

	example_batch, label_batch = tf.train.batch(  
		[example, label], batch_size=2)  
	#这个为内存队列，把reader从文件名队列读的东西放入这里
	#当名字队列的东西读完了之后，读到结束符号，则再从头读名字队列(而且新的队列也会重新打乱)
	#batch1
	# [B内容,label0] 
	# [A内容,label1]
	# [C内容,label2]
	return example_batch, label_batch
example_batch, label_batch=get_data()


with tf.Session() as sess:  
	coord = tf.train.Coordinator()  
	threads = tf.train.start_queue_runners(coord=coord) 
	#coord = tf.train.Coordinator()
	#tf.train.start_queue_runners(coord=coord)
	a,b=sess.run([example_batch,label_batch])
	print('循环前:',a,' |  ',b)
	for i in range(10):  #前面已经取了一次，后面就从第二个batch开始取
		a,b=sess.run([example_batch,label_batch])#a取得第一行batch个，
		print(a,' |  ',b)
		#print(a | b)
	coord.request_stop()  
	coord.join(threads)