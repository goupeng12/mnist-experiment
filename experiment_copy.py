import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 

# soft回归模型
x = tf.placeholder(tf.float32,[None,784]) # 60000x784的矩阵
W = tf.Variable(tf.zeros([784,10]))       # 784x10的矩阵
b = tf.Variable(tf.zeros([10]))           # 行向量
y = tf.nn.softmax(tf.matmul(x,W)+b)       # matmul(x,W) 结果为 60000x10 的矩阵，每行加上偏置行向量b
										  # 经softmax后，变为归一化的 60000x10 的概率矩阵

# 训练模型
y_ = tf.placeholder("float",[None,10])    # 60000x10的矩阵
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict = {x:batch_xs, y_ :batch_ys})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print (sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels}))