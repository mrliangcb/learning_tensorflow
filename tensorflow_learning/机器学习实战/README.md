# 机器学习实战  

参考文档:
https://github.com/machinelearningmindset/TensorFlow-Course/tree/master/docs/tutorials/1-basics/basic_math_operations  
https://github.com/nfmcclure/tensorflow_cookbook  
https://github.com/ahangchen/GDLnotes 
 
```py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
```

## 1.工作原理  
一般步骤: 通过placeholder输入数据; 给计算图提供数据; 评估损失函数; 反向传播  

## 2.创建张量，并进行运算  
```py

#直接定义计算模型   
v1 = tf.Variable(np.array(1,dtype=np.float64),name="v1")# dtype=tf.float32  
v11 = tf.cast(v1,dtype=tf.float32)#使用tf函数转化数据类型,64转成32  
v2 = tf.Variable(np.array(2,dtype=np.float32), name="v2")  
v3=v11+v2#两者要同样类型才能相加#模拟建立模型  

#函数定义模型，输入一个variable  
def custom_polynomial(x_val):  
		# 计算 3x^2 - x + 10  
		return(tf.subtract(3 * tf.square(x_val), x_val) + 10)#tf.square换成**2也行

global_init=tf.global_variables_initializer() #变量初始化定义
#单个变量相加（对于variable记得要run(init)）
with tf.Session() as sess:
	sess.run(global_init)
	print(sess.run(v1).dtype)
	print(sess.run(v11).dtype)
	print(sess.run(v3))
	print(sess.run(custom_polynomial(11)))

```















































