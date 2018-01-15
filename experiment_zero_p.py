"""
本实验在experiment.py的基础上，做了如下改动：
@使用了tensorboard

"""
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 

# soft回归模型

# 第一层
with tf.name_scope('input_layer') :
	x  = tf.placeholder(tf.float32,[None,784]) # 60000x784的矩阵
	y_ = tf.placeholder("float",[None,10])     # 60000x10 的矩阵

with tf.name_scope('hidden_layer1')	:
	with tf.name_scope('weight'):
		W1 = tf.Variable(tf.zeros([784,10]))  # 784x10的矩阵
		tf.summary.histogram('hidden_layer1/weight',W1)
	with tf.name_scope('weight'):
		b1 = tf.Variable(tf.zeros([10]))      # 长度为10的行矩阵
		tf.summary.histogram('hidden_layer1/weight',b1)
	with tf.name_scope('W1x1b1'):
		W1x1b1 = tf.matmul(x,W1)+b1 		  # matmul(x,W1) 为60000x10的矩阵，每行加上偏置行向量b1	
		tf.summary.histogram('hidden_layer/W1x1b1',W1x1b1)									      

y = tf.nn.softmax(W1x1b1) #激活函数,经softmax后，变为归一化的 60000x10 的概率矩阵


# 训练模型


# 用交叉熵作为损失函数
with tf.name_scope('loss'):
	loss = - tf.reduce_sum(y_ * tf.log(y))
	tf.summary.scalar('loss',loss)

with tf.name_scope('train'):	
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 评估模型
with tf.name_scope('estimation'):
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	tf.summary.scalar('accuracy',accuracy)

init = tf.initialize_all_variables()
with tf.Session() as sess :
	sess.run(init)
	merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
	train_writer = tf.summary.FileWriter('exp_zero_train_logs',sess.graph) #将训练日志写入到logs文件夹下
	# valid_writer = tf.summary.FileWriter('valid_logs',sess.graph) #将训练日志写入到logs文件夹下

	for i in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step,feed_dict = {x:batch_xs, y_ :batch_ys})
		if(i%500==0):
			result = sess.run(merged,feed_dict={x:batch_xs,y_:batch_ys}) #计算需要写入的日志数据
			#result1 = sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys}) 
			train_writer.add_summary(result,i) #将日志数据写入文件
			#valid_writer.add_summary(result1,i) #将日志数据写入文件
