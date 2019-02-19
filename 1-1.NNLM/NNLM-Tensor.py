# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

# 1. 合并所有句子，分割得到词包，不去重。
word_list = " ".join(sentences).split()
print(word_list)

# 2. 对词包去重
word_list = list(set(word_list))

# 3. 对每个词进行编号
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary
print(word_dict, n_class)

# NNLM Parameter
# 词向量特征数
m = 5
# 句长
n_step = 2 # number of steps ['i like', 'i love', 'i hate']
# 隐单元数量
n_hidden = 2 # number of hidden units

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        # 去除每一句的最后一个词，作为输入
        input = [word_dict[n] for n in word[:-1]]
        # 将最后一个词作为预测目标
        target = word_dict[word[-1]]

        # 将每个词转化为对应的激活向量，可激活词向量矩阵中的对应词向量
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model, y=b+U*tanh(d+h*x)
# 原论文中的公式为 y = b+ W*x + U*tanh(d+H*x)
# 当不希望x直接连结到输出时，W可以为零，这里就是省略了W*x
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

# 词向量矩阵
C = tf.Variable(tf.random_normal([1,n_class, m]))
C_shared = tf.tile(C, [tf.shape(X)[0],1,1]) # share parameter
vecs = tf.matmul(X,C_shared) # [batch_size, n_step, m], get freature vectors for every word

input = tf.reshape(vecs, shape=[-1, n_step * m]) # [batch_size, n_step * m]
b = tf.Variable(tf.random_normal([n_class]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
H = tf.Variable(tf.random_normal([n_step * m, n_hidden]))

tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

# softmax 层+ 交叉熵优化目标
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction =tf.argmax(model, 1) # 返回model向量中最大值的索引号

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

# 开始训练
for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
# 进行测试，这里直接用了训练集进行测试
# predict 的值是概率最大的词的索引
predict =  sess.run([prediction], feed_dict={X: input_batch})

# Test
#input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])
