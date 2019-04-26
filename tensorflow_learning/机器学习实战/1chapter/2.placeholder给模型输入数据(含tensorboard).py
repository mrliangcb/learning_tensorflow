import numpy as np
import tensorflow as tf

#placeholder其实就是一个variable，是一个图的起点

x = tf.placeholder(tf.float32, shape=(4, 4),name='input_place1') #place要设置 数据类型，大小
rand_array = np.random.rand(4, 4) #np矩阵
a=np.arange(0,16).reshape(4,4)  #np矩阵
y = tf.identity(x)#placeholder 输入为placeholder

sess = tf.Session()

print(sess.run(y, feed_dict={x: rand_array}))#给placeholder塞入一个np矩阵 

#将图保存
merged = tf.summary.merge_all()
path=r'C:/lcb/learning_python_git/learning_tensorflow/tensorflow_learning/机器学习实战/1chapter'
writer = tf.summary.FileWriter(path+"/1_tmp/variable_logs", sess.graph)


#调用tensorboard
#cmd先进入1_tmp 
# tensorboard --logdir=./logs #./variable_logs 直接读这个文件夹，不用读文件夹里面的文件
#Then open http://0.0.0.0:6006/ into your web browser
#   http://localhost:6006/    输入这个，不用上面那个
#上面一栏,scalars看监视tensor，graphs看图结构



# #定义完变量

# # 加入到tensorboard
# merged = tf.summary.merge_all()

# # 初始化写图的笔
# writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# # 初始化参数
# initialize_op = tf.global_variables_initializer()

# #运行初始化，tensorboard就会同步出现变化
# sess.run(initialize_op)