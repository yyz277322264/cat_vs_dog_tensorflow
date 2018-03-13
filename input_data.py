#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note:
# it is suggested to restart your kenel to train the model multiple times
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
train_dir = 'E:\yyz\python\Pig_rec/train/'


def get_files(file_dir):#查找数据存放路径
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    pig1 = []
    label1 = []
    pig2 = []
    label2 = []
    pig3 = []
    label3 = []
    pig4 = []
    label4 = []
    pig5 = []
    label5 = []
    pig6 = []
    label6 = []
    pig7 = []
    label7 = []
    pig8 = []
    label8 = []
    pig9 = []
    label9 = []
    pig10 = []
    label10 = []
    pig11 = []
    label11 = []
    pig12 = []
    label12 = []
    pig13 = []
    label13 = []
    pig14 = []
    label14 = []
    pig15 = []
    label15 = []
    pig16 = []
    label16 = []
    pig17 = []
    label17 = []
    pig18 = []
    label18 = []
    pig19 = []
    label19 = []
    pig20 = []
    label20 = []
    pig21 = []
    label21 = []
    pig22 = []
    label22 = []
    pig23 = []
    label23 = []
    pig24 = []
    label24 = []
    pig25 = []
    label25 = []
    pig26 = []
    label26 = []
    pig27 = []
    label27 = []
    pig28 = []
    label28 = []
    pig29 = []
    label29 = []
    pig30 = []
    label30 = []
    pig=[pig1,pig2,pig3,pig4,pig5,pig6,pig7,pig8,pig9,pig10,pig11,pig12,pig13,pig14,pig15,pig16,pig17,pig18,pig19,pig20,pig21,pig22,pig23,pig24,pig25,pig26,pig27,pig28,pig29,pig30]
    label=[label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11,label12,label13,label14,label15,label16,label17,label18,label19,label20,label21,label22,label23,label24,label25,label26,label27,label28,label29,label30]
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        pig[int(name[1])-1].append(file_dir + file)
        label[int(name[1])-1].append(name[1])

    # print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    pig_list=[]
    label_list = []
    for i in range(30):
        pig_list=np.hstack((pig_list,pig[i]))
        label_list = np.hstack((label_list, label[i]))
    print(pig_list)
    print(label_list)
    temp = np.array([pig_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i)-1 for i in label_list]
    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=capacity)

    #you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch



#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes


# testing

# import matplotlib.pyplot as plt
# #
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 50
# IMG_H = 50
#
# train_dir = 'E:\yyz\python\Pig_rec/train/'
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
# #%%






