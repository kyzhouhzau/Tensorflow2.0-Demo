#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow.keras.layers as layers
import time
(train_images,train_labels),(_,_) = tf.keras.datasets.mnist.load_data()

tf.enable_eager_execution()
#图片归一化
print(train_images.shape)
train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images = (train_images-127.5)/127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256

#实例化dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False))

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",use_bias=False,activation="tanh"))


    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),(2,2),padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

#实例化生成器
generator = make_generator_model()

#实例化判别器

discriminator = make_discriminator_model()

#定义生成损失

def generator_loss(generator_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generator_output),generator_output)

#定义判别损失
def dicriminator_loss(real_output,generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output),logits=real_output)
    generator_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),logits=generated_output)
    return real_loss+generator_loss

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

#定义检查点
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator=discriminator)

#

EPOCH = 50
NOISE_DIM = 100
num_examples_to_generator = 16

random_vector_for_generation = tf.random_normal([num_examples_to_generator,NOISE_DIM])



def train_step(images):
    noise = tf.random_normal([BATCH_SIZE,NOISE_DIM])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        generted_images = generator(noise,training=True)
        real_output = discriminator(images,training=True)

        generated_output = discriminator(generted_images,training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = dicriminator_loss(real_output,generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.variables))

#加快运算速度

train_step = tf.contrib.eager.defun(train_step)

def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def train(dataset,epoches):
    for epoch in range(epoches):
        start = time.time()
        for batch,images in enumerate(dataset):
            train_step(images)
            generate_and_save_images(generator,
                                 epoch + 1,
                                 random_vector_for_generation)
            print("BATCH:{}".format(batch))
        if (epoch+1)%15 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                             time.time() - start))
    generate_and_save_images(generator,
                             epoches,
                             random_vector_for_generation)
#训练模型
train(train_dataset,EPOCH)

























