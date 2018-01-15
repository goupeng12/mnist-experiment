import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一层:输入层
with tf.name_scope('input_layer') :
  x = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10])
  keep_prob = tf.placeholder("float")

# 第二层：卷积层+池化
with tf.name_scope('conv_layer1'):
  with tf.name_scope('weight'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    tf.summary.histogram('conv_layer1/weight',W_conv1)
  with tf.name_scope('bias'):
    b_conv1 = bias_variable([32])
    tf.summary.histogram('conv_layer1/bias',b_conv1)
  with tf.name_scope('x_image'):
    x_image = tf.reshape(x, [-1,28,28,1])
    tf.summary.histogram('conv_layer1/x_image',x_image)
  with tf.name_scope('h_conv1'):
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    tf.summary.histogram('conv_layer1/h_conv1',h_conv1)
  with tf.name_scope('h_pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram('conv_layer1/h_pool1',h_pool1)

# 第三层：卷积层+池化
with tf.name_scope('conv_layer2'):
  with tf.name_scope('weight'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    tf.summary.histogram('conv_layer2/weight',W_conv2)
  with tf.name_scope('bias'):
    b_conv2 = bias_variable([64])
    tf.summary.histogram('conv_layer2/bias',b_conv2)
  with tf.name_scope('h_conv2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    tf.summary.histogram('conv_layer2/h_conv2',h_conv2)
  with tf.name_scope('h_pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('conv_layer2/h_pool2',h_pool2)

# 第四层：dropout层
with tf.name_scope('dropout'):
  with tf.name_scope('weight'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    tf.summary.histogram('dropout/weight',W_fc1)
  with tf.name_scope('bias'):
    b_fc1 = bias_variable([1024])
    tf.summary.histogram('dropout/bias',b_fc1)
  with tf.name_scope('h_pool2_flat'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    tf.summary.histogram('dropout/h_pool2_flat',h_pool2_flat)
  with tf.name_scope('h_fc1'):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('dropout/h_fc1',h_fc1)
  with tf.name_scope('h_fc1_drop'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram('dropout/h_fc1_drop',h_fc1_drop)

# 第五层：dropout层
with tf.name_scope('fully-connect'):
  with tf.name_scope('weight'):
    W_fc2 = weight_variable([1024, 10])
    tf.summary.histogram('fully-connect/weight',W_fc2)
  with tf.name_scope('bais'):
    b_fc2 = bias_variable([10])
    tf.summary.histogram('fully-connect/bias',b_fc2)
  with tf.name_scope('y_conv'):
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    tf.summary.histogram('fully-connect/y_conv',y_conv)

with tf.name_scope('loss'):
  cross_entropy = tf.reduce_mean(tf.square(y_ - y_conv))
  tf.summary.scalar('loss/loss',cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('estimation'):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.summary.scalar('estimation/accuracy',accuracy)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
train_writer = tf.summary.FileWriter('exp_nine_logs',sess.graph) #将训练日志写入到logs文件夹下

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  if i%10 == 0:
    result = sess.run(merged,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) #计算需要写入的日志数据
    train_writer.add_summary(result,i) #将日志数据写入文件
  print(i)