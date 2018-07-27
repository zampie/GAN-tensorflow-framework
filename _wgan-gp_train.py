from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
from time import time
import models as models
import matplotlib.pyplot as plt

str_time = time()

max_epoch = 3000
batch_size = 64
lr = 0.0002
z_dim = 100
n_critic = 5
gpu_id = 0

#set out and ckpt
output_num_sqrt = 8
output_sample_per_it = 200
save_per_it = 1000

# set dataset
# input image size should be divisible by 16
dataset_name = "hy6"
image_type = "jpg"

# set dataset path
img_paths = glob.glob('./data/' + dataset_name + '/*.' + image_type)
input_image_shape = plt.imread(img_paths[0]).shape
output_image_size = input_image_shape[0]

#set save path
index = '0'
GAN_type = "wgan-gp"



# sample_path = './samples/' + dataset_name + '_' + GAN_type + '_' + index
sample_path = 'C:/Users/Zampie/OneDrive/train/samples/' + dataset_name + '_' + GAN_type + '_' + index

log_path = './logs/' + dataset_name + '_' + GAN_type + '_' + index
ckpt_path = './checkpoints/' + dataset_name + '_' + GAN_type + '_' + index
utils.mkdir(sample_path + '/')
utils.mkdir(ckpt_path + '/')

def preprocess_fn(img):
    # re_size = 64
    # img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    img = tf.to_float(img) / 127.5 - 1

    return img

data_pool = utils.DiskImageData(img_paths, batch_size, shape=input_image_shape, preprocess_fn=preprocess_fn)

with tf.device('/gpu:%d' % 0):
    generator = models.generator
    discriminator = models.discriminator_wgan_gp

    # inputs
    real = tf.placeholder(tf.float32, shape=[None, input_image_shape[0], input_image_shape[1], input_image_shape[2]])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # generate
    fake = generator(z, size=output_image_size, reuse=False)

    # dicriminate
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        # update
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=list(range(1, x.shape.ndims))))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, size=output_image_size, training=False)


# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter(log_path, sess.graph)

if not utils.load_checkpoint(ckpt_path, sess):
    sess.run(tf.global_variables_initializer())

try:
    z_ipt_sample = np.random.normal(size=[output_num_sqrt * output_num_sqrt, z_dim])


    # batch_epoch = len(data_pool) // (batch_size * n_critic)

    # generator epoch
    batch_epoch = len(data_pool) // batch_size

    max_it = max_epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        # it_epoch = it % batch_epoch + 1
        it_epoch = it % batch_epoch

        # train D
        for i in range(n_critic):
            # batch data
            real_ipt = data_pool.batch()
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            cost_time = time() - str_time
            print("Epoch:%4d  Iteration:%5d   Batch:%3d/%3d   Time:%.3f" % (epoch, it, it_epoch, batch_epoch, cost_time))

        # save
        if (it + 1) % save_per_it == 0 or it == max_it - 1:
            save_name = "Epoch_" + str(epoch).zfill(4) + "_It_" + str(it).zfill(5)
            save_path = saver.save(sess, ckpt_path + "/" + save_name)
            print('Checkpoint saved in: % s' % save_path)

        # sample
        if (it + 1) % output_sample_per_it == 0 or it == 0 or it == max_it - 1:
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})

            # sample_name = "Epoch_" + str(epoch).zfill(4) + "_It_" + str(it).zfill(5) + "_" + str(it_epoch) + "of" + str(batch_epoch)
            sample_name = "Epoch_" + str(epoch).zfill(4) + "_It_" + str(it).zfill(5)

            utils.imwrite(utils.immerge(f_sample_opt, output_num_sqrt, output_num_sqrt), sample_path + "/" + sample_name + ".jpg")
            print('Sample saved in: % s' % sample_path)




except Exception as e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
