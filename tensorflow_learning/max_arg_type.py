#求最大概率的类
import numpy as np

#模型结构
#定义模型的时候多写一个模型，输入测试数据loader
# pred = tf.nn.softmax(tf.matmul(batch_xs, W) + b) # Softmax  x1*w1+x2*w2+……+b
# pred_test = tf.nn.softmax(tf.matmul(测试数据, W) + b)

# p=sess.run(pred_test)#输出np型，行为batch个数据，列为输出向量



# pred_y = np.append(pred_y, np.argmax(logits_value_test, axis=1))
					# real_y = np.append(real_y, _test_label)
a=np.array([[1,3,2],
			[4,1,3]])
b=np.argmax(a, axis=1)
d=np.array([1,1])
print(b)#[1 0]返回np
e=np.sum(a==b)#对比a和b有多少个相同
print(e)

e=np.append(a,b)

print(e)