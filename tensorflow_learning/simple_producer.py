#simple_producer
import tensorflow as tf  
filenames = [r'C:\lcb\learning_python_git\learning_tensorflow\tensorflow_learning\test.csv',r'./test2.csv'] 

filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
print(filename_queue)

# with tf.Session() as sess: 
	# coord = tf.train.Coordinator()  
	# threads = tf.train.start_queue_runners(coord=coord) 
	# print('输出:',sess.run(a))
	