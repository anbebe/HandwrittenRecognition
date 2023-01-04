import numpy as np
import os
from os import listdir
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
#import cv2
import random


def load_data(train_summary_writer, val_summary_writer, batch_size=32, visualise=True):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = pathlib.Path(dir_path + "/data/small_cropped/")

    batch_size = batch_size
    img_height = 32
    img_width = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print("classes: ", class_names)

    if visualise:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            print("Shape: ", images[0].shape)
            print("Label: ", labels[0], "->", class_names[labels[0]])
            with train_summary_writer.as_default():
                tf.summary.image("32 training data examples", images.numpy().astype(np.uint8), max_outputs=32, step=0)

    return train_ds, val_ds

def preprocess_data(data):
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    data = data.map(lambda x, t: ((x/255.), t))
    data = data.map(lambda x, t: (x, tf.one_hot(t, depth=6)))
    data = data.cache()
    data = data.shuffle(1000)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return data

def create_cropped_dataset():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = dir_path + "/data/small/"
    dst_dir = dir_path + "/data/small_cropped/"

    for label in ["boxing", "five", "jump", "quickly", "the", "wizards"]:
        path = data_dir + label
        print("path: ", path)
        for f in os.listdir(path):
            img_dir = path + "/" + f 
            print(img_dir)
            img = cv2.imread(img_dir)

            borderType = cv2.BORDER_CONSTANT
            value = [255,255,255]
            top = 40
            bottom = top
            left = 7
            right = left

            dst = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)
            stretch_near = cv2.resize(dst, (32, 32), interpolation = cv2.INTER_AREA)
            dst_dir_img = dst_dir + label + "/" + f 
            cv2.imwrite(dst_dir_img, stretch_near)


def create_extended_cropped_dataset():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = dir_path + "/data/small_cropped/"
    dst_dir = dir_path + "/data/small_extended_cropped/"

    for label in ["boxing", "five", "jump", "quickly", "the", "wizards"]:
        path = data_dir + label
        print("path: ", path)
        for f in os.listdir(path):
            img_dir = path + "/" + f 
            print(img_dir)
            img = cv2.imread(img_dir)
            
            angles = range(-30,30)
            angle = random.choice(angles)
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            contrasted_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            
            dst_dir_img = dst_dir + label + "/aug_" + f
            print(dst_dir_img)
            cv2.imwrite(dst_dir_img, contrasted_img)    
