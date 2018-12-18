
import sys
import os
from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorlayer as tl
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50


class Generator:
    def __init__(self):
        pass

    def build(self, images, labels, z):
        input_size = labels.shape[1].value + z.get_shape()[1].value
        output_size = images.shape[1].value

        num_hidden_nodes = 1000
        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('generator'):
            self.G_W1 = tf.Variable(
                xavier_init([input_size, num_hidden_nodes]), name='G_W1')
            self.G_b1 = tf.Variable(
                tf.zeros(shape=[num_hidden_nodes]), name='G_b1')

            self.G_W2 = tf.Variable(
                xavier_init([num_hidden_nodes, output_size]), name='G_W2')
            self.G_b2 = tf.Variable(
                tf.zeros(shape=[output_size]), name='G_b2')

            self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

    def generate(self, z, labels):
        inputs = tf.concat(axis=1, values=[z, labels])

        with tf.variable_scope('generator', reuse=True):
            G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
            G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob


class Discriminator:
    def __init__(self):
        pass

    def build(self, images, labels):
        input_size = images.shape[1].value + labels.shape[1].value
        output_size = 1

        num_hidden_nodes = 1000
        xavier_init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('discriminator'):
            self.D_W1 = tf.Variable(
                xavier_init([input_size, num_hidden_nodes]), name='D_W1')
            self.D_b1 = tf.Variable(
                tf.zeros(shape=[num_hidden_nodes]), name='D_b1')

            self.D_W2 = tf.Variable(
                xavier_init([num_hidden_nodes, output_size]), name='D_W2')
            self.D_b2 = tf.Variable(
                tf.zeros(shape=[output_size]), name='D_b2')

            self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def discriminate(self, images, labels):
        inputs = tf.concat(axis=1, values=[images, labels])

        with tf.variable_scope('discriminator', reuse=True):
            D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
            D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
            D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


class Cgan:
    def __init__(self):
        pass

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

        # NOTE: negative sign on loss because we want to maximize not minimize
        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        # d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        # g_vars = tl.layers.get_variables_with_name('generator', True, True)

        self.D_solver = tf.train.AdamOptimizer(learning_rate=0.001) \
            .minimize(self.D_loss, var_list=self.discriminator.theta_D)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=0.001) \
            .minimize(self.G_loss, var_list=self.generator.theta_G)

    def run(self, n_epochs=5000, output=None, every=10):
        sess = tf.Session()

        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))
         
        print('Initial theta_G:', sess.run(self.generator.theta_G))

        for epoch in range(0, n_epochs + 1):
            _, D_loss_curr = sess.run([self.D_solver, self.D_loss])
            _, G_loss_curr = sess.run([self.G_solver, self.G_loss])

            print('epoch:', epoch, 'theta_G', sess.run(self.generator.theta_G))

            # if epoch % 10 == 0:
            #     print(epoch)
            #     save_imgs(epoch, sample[: 25], output)

            if epoch % every == 0:
                print(epoch, D_loss_curr, G_loss_curr)
                sample = sess.run(self.sample)
                save_imgs(epoch, sample[: 25], output)

        sess.close()


def save_imgs(epoch, sample, output):
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            print(sample[cnt].reshape(256, 256))
            axs[i, j].imshow(sample[cnt].reshape(256, 256), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("{}/album_{}.png".format(output, epoch))
    plt.close()


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
        images = list(map(lambda file: np.array(Image.open(file).convert('L')).reshape(-1),
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

    dataset = (np.array(images_ordered).astype(np.float32),
               np.array(embeddings_ordered).astype(np.float32))

    print('Dataset loaded. Size:', dataset[0].shape, dataset[1].shape)

    return dataset, output

if __name__ == '__main__':
    main()
256