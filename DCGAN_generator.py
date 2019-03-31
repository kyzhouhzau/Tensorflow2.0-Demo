#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from DCGAN import *

#重载模型并生成图像

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open(PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no)))

display_image(EPOCH)