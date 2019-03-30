#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
文本生成
#https://www.tensorflow.org/tutorials/sequences/text_generation
"""
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
#下载数据
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#读取数据
text = open(path_to_file).read()
print("Length of text:{} characters".format(len(text)))

vocab = sorted(set(text))

char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


#只想打印出20个
for char,_ in zip(char2idx,range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char),char2idx[char]))
print ('{} ---- characters mapped to int ---- > {}'.format(text[:13], text_as_int[:13]))

seq_length=100
#将文本拆分成文本块

print(text_as_int.shape)
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1,drop_remainder=True)

for item in chunks.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_input = chunk[1:]
    return input_text,target_input

dataset = chunks.map(split_input_target)

for input_examlpe,target_example in dataset.take(1):
    print("Input data:",repr("".join(idx2char[input_examlpe.numpy()])))
    print("Ouput data:",repr("".join(idx2char[target_example.numpy()])))


BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

#定义模型，一共三层。
#想用与训练好的词向量。
from tensorflow.python.keras.engine.base_layer import Layer
class Embedding(Layer):
    #也可以通过tf.keras.layers.Embedding 的weight参数传入

    def __init__(self,in_size,out_size,pretrained=None,**kwargs):
        super(Embedding,self).__init__(**kwargs)
        if pretrained==None:
            self.embeddings = tf.Variable(tf.random_uniform([in_size, out_size]))
        else:
            self.embeddings = tf.Variable(np.array(pretrained))
    def call(self, inputs):
        inputs = math_ops.cast(inputs, 'int32')
        return embedding_ops.embedding_lookup(self.embeddings,inputs)

class Model(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,units):
        super(Model,self).__init__()
        self.units = units
        #keras中的使用方法
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
        #
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                recurrent_initializer="glorot_uniform",
                                                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           recurrent_activation='sigmoid',
                                           stateful=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self,x):
        embedding = self.embedding(x)

        oupt = self.gru(embedding)
        prediction = self.fc(oupt)
        return prediction

VOCAB_SIZE=len(vocab)
EMBEDDING_DIM = 256
UNITS = 1024

model = Model(VOCAB_SIZE,EMBEDDING_DIM,UNITS)

#创建优化器
optimizer = tf.train.AdamOptimizer()
def loss_function(real,preds):
    #用sparse_softmax_cross_entropy不需要用one-hot编码
    return tf.losses.sparse_softmax_cross_entropy(labels=real,logits=preds)
##############################
#这里使用model.build（）是为了告诉模型我们输入的数据维度
model.build(tf.TensorShape([BATCH_SIZE,seq_length]))
#打印所有参数
model.summary()
##############################

checkpoint_dir = "./traing_checkpoit"
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")

EPOCHS=5
#
#
#
#
#训练模型
loss=0
for epoch in range(EPOCHS):
    start = time.time()
    #这里注意，初始化颖仓曾初始状态
    #注意这里我们每一个epoch初始化一次
    hidden = model.reset_states()
    for (batch ,(inp,target)) in enumerate(dataset):
        #tf.Gradientape记录计算过程
        with tf.GradientTape() as tape:
            predictions = model(inp)
            loss = loss_function(target,predictions)
        #loss 对所有变量求导
        grads = tape.gradient(loss,model.variables)
        #自动更新模型参数
        optimizer.apply_gradients(zip(grads,model.variables))


        print("EPOCH {} Batch {} Loss {:.4f}".format(epoch+1,batch,loss))

    #保存检查点
    if (epoch +1) % 5 ==0:
        model.save_weights(checkpoint_prefix)
    print("Epoch {} Loss {.4f}".format(epoch+1,loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model = Model(VOCAB_SIZE,EMBEDDING_DIM,UNITS)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))


#Evaluation loop

num_generate = 1000
start_string = 'Q'

input_eval = [char2idx[s] for s in start_string]#[10]
input_eval = tf.expand_dims(input_eval,0)#[[10]]
text_generated = []

temperature = 1.0



model.reset_states()
for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions,0)
    predictions = predictions/temperature
    predicted_id = tf.multinomial(predictions,num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id],0)

    text_generated.append(idx2char[predicted_id])
print(start_string+''.join(text_generated))

