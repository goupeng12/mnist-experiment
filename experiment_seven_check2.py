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

"""
# 第二层：卷积层+池化
with tf.name_scope('conv_layer1'):
  with tf.name_scope('weight'):
    W_conv1 = weight_variable([5, 5, 1, 32]) # 5x5的卷积核，单通道，32个卷积核
    tf.summary.histogram('conv_layer1/weight',W_conv1)
  with tf.name_scope('bias'):
    b_conv1 = bias_variable([32])            #32个偏置值
                                             #一张28x28的图像与上述卷积核卷积后的结果为32个28x28的矩阵
                                             #每个矩阵一个偏置值，偏置值将会加到这个矩阵的每一个元素上面
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
  # 第一层操作的结果  图片张数(batch) 个 28x28的矩阵，32通道
  # 池化结果 图片张数(batch) 个 14x14的矩阵，32通道

# 第三层：卷积层+池化
with tf.name_scope('conv_layer2'):
  with tf.name_scope('weight'):
    W_conv2 = weight_variable([5, 5, 32, 64]) # 5x5的卷积核，32通道，64个卷积核
    tf.summary.histogram('conv_layer2/weight',W_conv2)
  with tf.name_scope('bias'):
    b_conv2 = bias_variable([64])             # 32个偏置值
                                              # 一张14x14 32通道的图像与上述卷积核卷积后的结果为64个14x14的矩阵
    tf.summary.histogram('conv_layer2/bias',b_conv2)
  with tf.name_scope('h_conv2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    tf.summary.histogram('conv_layer2/h_conv2',h_conv2)
  with tf.name_scope('h_pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('conv_layer2/h_pool2',h_pool2)
  # 第二层操作的结果  图片张数(batch) 个 14x14的矩阵，64通道
  # 池化结果 图片张数(batch) 个 7x7的矩阵，64通道

# 第三层：卷积层+池化
with tf.name_scope('conv_layer3'):
  with tf.name_scope('weight'):
    W_conv3 = weight_variable([5, 5, 64, 128])
    tf.summary.histogram('conv_layer2/weight',W_conv3)
  with tf.name_scope('bias'):
    b_conv3= bias_variable([128])
    tf.summary.histogram('conv_layer2/bias',b_conv3)
  with tf.name_scope('h_conv3'):
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    tf.summary.histogram('conv_layer3/h_conv3',h_conv3)
  #with tf.name_scope('h_pool3'):
   # h_pool3 = max_pool_2x2(h_conv3)
    #tf.summary.histogram('conv_layer3/h_pool3',h_pool3)
  # 第三层操作的结果   图片张数(batch) 个 14x14的矩阵，128通道
  # 池化结果：不做卷积 图片张数(batch) 个 7x7的矩阵，128通道

# 第四层：dropout层
with tf.name_scope('dropout'):
  with tf.name_scope('weight'):
    W_fc1 = weight_variable([7 * 7 * 128, 1024])
    tf.summary.histogram('dropout/weight',W_fc1)
  with tf.name_scope('bias'):
    b_fc1 = bias_variable([1024])
    tf.summary.histogram('dropout/bias',b_fc1)
  with tf.name_scope('h_pool2_flat'):
    h_pool2_flat = tf.reshape(h_conv3, [-1, 7*7*128])
    tf.summary.histogram('dropout/h_pool2_flat',h_pool2_flat)
  with tf.name_scope('h_fc1'):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('dropout/h_fc1',h_fc1)
  with tf.name_scope('h_fc1_drop'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram('dropout/h_fc1_drop',h_fc1_drop)

# 第五层：fully-connect层
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
  cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
  tf.summary.scalar('loss/loss',cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('estimation'):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.summary.scalar('estimation/accuracy',accuracy)
"""

sess = tf.InteractiveSession()

saver = tf.train.Saver() # 生成 saver
saver.restore(sess,"exp_seven_model_savepath/")

print("start")
batch = mnist.train.next_batch(50)
print (sess.run(accuracy,feed_dict = {x:batch[0],y_:batch[1],keep_prob: 0.5}))

print("end")

