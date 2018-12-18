
import sys
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import *
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 128


class Generator:
    def __init__(self):
        pass

    def build(self, images, labels, z):
        self.input_size = labels.shape[1].value + z.get_shape()[1].value
        self.output_size = images.shape[1].value
        self.label_size = labels.shape[1].value

    def generate(self, z, labels,
                 is_train=True, reuse=False, batch_size=BATCH_SIZE):
        """ z + (txt) --> 64x64 """
        # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
        s = self.output_size  # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = 128
        self.num_channels = 3

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1.0, 0.02)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                net_in = InputLayer(z, name='g_inputz')
                net_txt = InputLayer(labels, name='g_input_txt')

                net_txt = DenseLayer(net_txt, n_units=self.label_size,
                        act=lambda x: tl.act.lrelu(x, 0.2), 
                        W_init=w_init, name='g_reduce_text/dense')
                net_in = ConcatLayer([net_in, net_txt], concat_dim=1,
                                     name='g_concat_z_txt')

                net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16,
                                    act=tf.identity,
                                    W_init=w_init,
                                    b_init=None,
                                    name='g_h0/dense')
                net_h0 = BatchNormLayer(net_h0,
                                        # act=tf.nn.relu,
                                        is_train=is_train, 
                                        gamma_init=gamma_init,
                                        name='g_h0/batch_norm')
                net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8],
                                      name='g_h0/reshape')

                net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                        padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
                net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                        gamma_init=gamma_init, name='g_h1_res/batch_norm')
                net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
                net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                        gamma_init=gamma_init, name='g_h1_res/batch_norm2')
                net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
                net = BatchNormLayer(net, # act=tf.nn.relu,
                        is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
                net_h1 = ElementwiseLayer([net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
                net_h1.outputs = tf.nn.relu(net_h1.outputs)

                # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
                # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
                #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
                net_h2 = UpSampling2dLayer(net_h1, size=( s8, s8 ), is_scale=False, method=1,
                        align_corners=False, name='g_h2/upsample2d')
                net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
                net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
                        is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

                net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                        padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
                net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                        gamma_init=gamma_init, name='g_h3_res/batch_norm')
                net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
                net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                        gamma_init=gamma_init, name='g_h3_res/batch_norm2')
                net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
                net = BatchNormLayer(net, #act=tf.nn.relu,
                        is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
                net_h3 = ElementwiseLayer([net_h2, net], combine_fn=tf.add, name='g_h3/add')
                net_h3.outputs = tf.nn.relu(net_h3.outputs)

                net_h4 = UpSampling2dLayer(net_h3, size=( s4, s4 ), is_scale=False, method=1,
                        align_corners=False, name='g_h4/upsample2d')
                net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
                net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                        is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

                net_h5 = UpSampling2dLayer(net_h4, size=(s2, s2), is_scale=False, method=1,
                        align_corners=False, name='g_h5/upsample2d')
                net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
                net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                        is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

                net_ho = UpSampling2dLayer(net_h5, size=(s, s), is_scale=False, method=1,
                        align_corners=False, name='g_ho/upsample2d')
                net_ho = Conv2d(net_ho, self.num_channels, (3, 3), (1, 1),
                        padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
                logits = net_ho.outputs
                net_ho.outputs = tf.nn.tanh(net_ho.outputs)

        return net_ho.outputs


class Discriminator:
    def __init__(self):
        pass

    def build(self, images, labels):
        self.image_size = images.shape[1].value
        self.labels_size = labels.shape[1].value
        input_size = images.shape[1].value + labels.shape[1].value
        output_size = 1

    def discriminate(self, input_images, t_txt, is_train=True):
        """ 64x64 + (txt) --> real/fake """
        # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
        # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        df_dim = 64  # 64 for flower, 196 for MSCOCO
        s = self.image_size  # output image size [64]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            net_in = InputLayer(input_images, name='d_input/images')
            net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                    padding='SAME', W_init=w_init, name='d_h0/conv2d')

            net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                    padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
            net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
            net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                    padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
            net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
            net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                    padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
            net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

            net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                    padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
            net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
            net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                    padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
            net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
            net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                    padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
            net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
            net_h4 = ElementwiseLayer([net_h3, net], combine_fn=tf.add, name='d_h4/add')
            net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

            if t_txt is not None:
                net_txt = InputLayer(t_txt, name='d_input_txt')
                net_txt = DenseLayer(net_txt, n_units=self.labels_size,
                    act=lambda x: tl.act.lrelu(x, 0.2),
                    W_init=w_init, name='d_reduce_txt/dense')
                net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
                net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
                # NOTE: 4 x 4 is hardcoded for the 64 by 64 images dataset
                net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
                net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
                # 243 (ndf*8 + 128 or 256) x 4 x 4
                net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                        padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
                net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                        is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

            net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
            # 1 x 1 x 1
#             net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
            logits = net_ho.outputs
            net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

        return net_ho.outputs, logits
    
class Cgan:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_

    def build(self, dataset, z_size=100):
        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next()

        self.X = next_batch[0]
        self.y = next_batch[1]
        self.Z = tf.random.uniform((BATCH_SIZE, z_size), -1, 1)

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator.build(self.X, self.y, self.Z)
        self.discriminator.build(self.X, self.y)

        self.sample = self.generator.generate(self.Z, self.y)
        D_real, D_logit_real = self.discriminator.discriminate(self.X, self.y)
        D_fake, D_logit_fake = self.discriminator.discriminate(self.sample,
                                                               self.y)

        self.d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        self.g_vars = tl.layers.get_variables_with_name('generator', True, True)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self.lambda_ = tf.constant(self.lambda_, dtype=tf.float32)
            d_losses = [tf.nn.l2_loss(tf.cast(input_, tf.float32)) for input_ in self.d_vars]
            g_losses = [tf.nn.l2_loss(tf.cast(input_, tf.float32)) for input_ in self.g_vars]

            self.d_losses = tf.add_n(d_losses)
            self.g_losses = tf.add_n(g_losses)

            # NOTE: negative sign on loss because we want to maximize not minimize
            self.D_loss = self.lambda_ * self.d_losses - tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            self.G_loss = self.lambda_ * self.g_losses - tf.reduce_mean(tf.log(D_fake))

            D_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.gvs = D_optimizer.compute_gradients(self.D_loss, var_list=self.d_vars)
            capped_gvs = [(tf.clip_by_value(grad, -10, 10.), var) for grad, var in self.gvs]
            self.D_solver = D_optimizer.apply_gradients(capped_gvs)

            G_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.gvs = G_optimizer.compute_gradients(self.G_loss, var_list=self.g_vars)
            capped_gvs = [(tf.clip_by_value(grad, -10, 10.), var) for grad, var in self.gvs]
            self.G_solver = G_optimizer.apply_gradients(capped_gvs)

    def run(self, n_epochs=5000, output=None, every=10):
        sess = tf.Session()

        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))

        for epoch in range(0, n_epochs + 1):
            _, D_loss_curr = sess.run([self.D_solver, self.D_loss])
            _, G_loss_curr = sess.run([self.G_solver, self.G_loss])
            
            print(f'Epoch {epoch}: Discriminator Loss: {D_loss_curr}, Generator Loss: {G_loss_curr}.')

            if epoch % every == 0:
                sample = sess.run(self.sample)
                now = datetime.now()
                new_folder = f'{output}/model_{epoch}_{now.day}_{now.hour}_{now.minute}_{now.second}'
                self.save_imgs(epoch, sample, new_folder)
                self.save_model(epoch, sess, new_folder)

        sess.close()


    def save_model(self, epoch, sess, output):
        new_model_folder = output + '/model'

        os.makedirs(new_model_folder, exist_ok=True)

        save_file = new_model_folder + '/model'
        saver = tf.train.Saver()
        saver.save(sess, save_file, global_step=epoch)
        print('Saved model to', save_file)


    def save_imgs(self, epoch, sample, output):
        new_images_folder = output + '/images'

        os.makedirs(new_images_folder, exist_ok=True)

        for i, picture in enumerate(sample):
            fig = plt.figure()
            ax = fig.gca()

            ax.imshow(picture.reshape(self.X.shape[1], self.X.shape[2], self.generator.num_channels))
            ax.axis('off')

            plt.savefig(new_images_folder + f'/generated_image_{i}.png', bbox_inches='tight')

            plt.close(fig)

        print('Saved images to', new_images_folder)


def main():
    dataset, output = parse_args()

    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)

    cgan = Cgan()
    cgan.build(dataset)
    cgan.run(n_epochs=5000, output=output)


def parse_args():
    arg_parser = ArgumentParser()

    arg_parser.add_argument('--images', required=True)
    arg_parser.add_argument('--text_embeddings', required=True)
    arg_parser.add_argument('--output', default='./tmp')

    args = arg_parser.parse_args()

    output = args.output

    if not os.path.isdir(output):
        os.mkdir(output)

    albums = {}

    for root, _, files in os.walk(args.images):
        img_files = list(filter(lambda file: file.endswith('.jpg'), files))
        keys = list(map(lambda file: file.strip('.jpg'), img_files))
        img_files = list(map(lambda file: os.path.join(root, file), img_files))
        images = list(map(lambda file: np.array(Image.open(file)),
                                                img_files))

        new_albums = {key: image for key, image in zip(keys, images)}

        albums.update(new_albums)

    with open(args.text_embeddings, 'rb') as txt_file:
        text_embeddings = pickle.load(txt_file)

    embeddings_ordered = list()
    images_ordered = list()

    albums_copy = list(albums)

    for album in albums_copy:
        try:
            embeddings_ordered.append(text_embeddings[album])
            images_ordered.append(albums[album])
        except KeyError:
            print(album, 'not found in embeddings.pkl. Removing', album, 'from dataset')

    images = np.array(images_ordered).astype(np.float32)
    embeddings = np.array(embeddings_ordered).astype(np.float32)
    embeddings = np.apply_along_axis(lambda embedding: embedding / np.std(embedding), 1, embeddings)
    embeddings = embeddings + np.abs(np.min(embeddings))

    dataset = (images / np.max(images), embeddings)

    print('Dataset loaded. Size:', dataset[0].shape, dataset[1].shape)

    return dataset, output


if __name__ == '__main__':
    main()
