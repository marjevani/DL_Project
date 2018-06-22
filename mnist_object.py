# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import numpy
import tensorflow as tf
import sys
import urllib
import os.path
import numpy as np
from datetime import datetime
from util import *

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import pickle

LOGDIR = 'logs_new_model_v04/'
GITHUB_URL = 'https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'


class Net:
    def __init__(self):
        self.resume = os.path.isdir(LOGDIR)
        ### MNIST EMBEDDINGS ###
        self.mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

        ### Get a sprite and labels file for the embedding projector  ###
 #       urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
 #       urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')


    # Add convolution layer
    @staticmethod
    def conv_layer(input, size_in, size_out, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            normed_out= tf.layers.batch_normalization(conv + b)
            act = tf.nn.relu(normed_out)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return conv

    @staticmethod
    def pooling_layer(input, name="pool"):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    @staticmethod
    def drop_out_layer(input,prob, name="drop"):
        with tf.name_scope(name):
            return tf.nn.dropout(x=input,keep_prob=prob,name=name)

    @staticmethod
    # Add fully connected layer
    def fc_layer(input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
            act = tf.nn.relu(tf.matmul(input, w) + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return act


    def mnist_model(self,learning_rate, use_two_conv, use_two_fc, hparam):
        tf.reset_default_graph()

        # Setup placeholders, and reshape the data
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

        # create conv layers and connect them:
        conv1_a = self.conv_layer(x_image, 1, 32, "conv1_a")
        conv1_b = self.conv_layer(conv1_a, 32, 32, "conv1_b")
        conv1_C = self.conv_layer(conv1_b, 32, 32, "conv1_c")
        pool1 = self.pooling_layer(conv1_C ,"pool1" )

        conv2_a = self.conv_layer(pool1, 32, 64, "conv2_a")
        conv2_b = self.conv_layer(conv2_a, 64, 64, "conv2_b")
        conv2_C = self.conv_layer(conv2_b, 64, 64, "conv2_c")
        pool2 =   self.pooling_layer(conv2_C ,"pool2" )


        flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
        fc1 = self.fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        self.prob =  tf.Variable(0.4, name="prob")
        dp = self.drop_out_layer(fc1,self.prob,"drop")
        embedding_input = fc1
        embedding_size = 1024
        self.logits = self.fc_layer(dp, 1024, 10, "fc2")


        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.y), name="cross_entropy")
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
       # print(tf.reduce_sum(tf.exp(self.logits), 1))
        #self.data =  tf.reduce_sum(tf.exp(self.logits), 1)#tf.exp(self.logits) / tf.reduce_sum(tf.exp(self.logits), 1)
        with tf.name_scope("accuracy"):

            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        self.summ = tf.summary.merge_all()

        self.embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
        self.assignment = self.embedding.assign(embedding_input)

        # enable GPU
        config = tf.ConfigProto()
        debug_print( "don't" if not tf.device('/gpu:0') else "" + "recognized GPU" )
        with tf.device('/gpu:0'):
                config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

        if(self.resume):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(LOGDIR))
        else:
            self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(LOGDIR)
        self.writer.add_graph(self.sess.graph)


        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = self.embedding.name
        embedding_config.sprite.image_path = 'sprite_1024.png'
        embedding_config.metadata_path = 'labels_1024.tsv'
        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([28, 28])
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.writer, config)

    def train(self):
        if os.path.exists(os.path.join(LOGDIR,"step")):
            with open(os.path.join(LOGDIR, "step"), 'rb') as file:
                step = pickle.load(file)
        else:
            step=0
        for i in range(6000*2):
            batch = self.mnist.train.next_batch(100)
            ## save statistics for tensorBoard
            if (i+step) % 5 == 0:
                self.prob.assign(0.4)
                [train_accuracy, s ,results] = self.sess.run([self.accuracy, self.summ , self.logits], feed_dict={self.x: batch[0], self.y: batch[1]})
               # print(results)
                self.writer.add_summary(s, (i+step))
            if (i+step) % 500 == 0:
                self.prob.assign(1)
                self.sess.run(self.assignment, feed_dict={self.x: self.mnist.test.images[:1024], self.y: self.mnist.test.labels[:1024]})
                self.saver.save(self.sess, os.path.join(LOGDIR, "model.ckpt"), (i+step))
                with open(os.path.join(LOGDIR, "step"), 'wb') as file:
                    pickle.dump((i+step),file)
            ## Train
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y: batch[1]})

    def eval(self,img):
        self.prob.assign(1)
        [train_accuracy, logits,] = self.sess.run([self.accuracy, self.logits], feed_dict={self.x: [img], self.y: [[0]*10]})
        # debug_print(numpy.exp(logits))
        debug_print(logits)
        eval_list = logits.tolist()[0]
        eval_val = eval_list.index(max(eval_list))
        sum_list = (sum(eval_list))
        eval_list_percentage=[]
        #eval_list_percentage= [ (x/sum_list)*100 for x in eval_list ]
        for i in range(len(eval_list)):
           eval_list_percentage.append((eval_list[i] / sum_list)*100)
        index=0
        for num in eval_list_percentage:
          print('Digit', index,':',round(num,2))
          index += 1

        print("The number is: " + str(eval_val))


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    # start time performance measure
    start_time = datetime.now()
    debug_print("starting training performance measure:")
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                debug_print('Starting run for %s' % hparam)

                # Actually run with the new settings
                net = Net()
                net.mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)
                net.train()

    end_time = datetime.now()
    debug_print('Training Duration: {}'.format(end_time - start_time))

def eval(img):
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                debug_print('Starting run for %s' % hparam)
                reshaped_img = []
                # Actually run with the new settings
                net = Net()
                for row in img:
                    for col in row:
                        reshaped_img.append(col)

                net.mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)
                net.eval(reshaped_img)


if __name__ == '__main__':
    main()