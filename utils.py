import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from PIL import Image

class ImageData:

    def __init__(self, img_size, channels, augment_flag=False):
        self.img_size = img_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        w = 178
        h = 218
        image = tf.image.resize_images(x_decode, [w, h])
        #img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        seed = random.randint(0,2 ** 31 -1)
        image = tf.image.random_flip_left_right(image, seed=seed)
        img = tf.image.resize_image_with_crop_or_pad(image,178,178)
        img = tf.image.resize_images(img,[self.img_size,self.img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        

        #if self.augment_flag :
        #    if self.img_size < 256 :
        #        augment_size = 256
        #    else :
        #        augment_size = self.img_size + 30
        #    p = random.random()
        #    if p > 0.5:
        #        img = augmentation(img, augment_size)

        return img

def cropimread(crop, xcrop, ycrop, img_pre):
    "Function to crop center of an image file"
    if crop:
        ysize, xsize, chan = img_pre.shape
        xoff = (xsize - xcrop) // 2
        yoff = (ysize - ycrop) // 2
        img= img_pre[yoff:-yoff,xoff:-xoff]
    else:
        img= img_pre
    return img\

def pilimage_crop(image_path,new_width,new_height):
    img = Image.open(image_path)
    width, height = img.size   # Get dimensions
   
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    print(left,top,right,bottom)
    return img.crop((left, top, right, bottom)).resize((256,256))

def crop_center(img,cropx,cropy):
    y,x,ch = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def load_test_data(image_path, size=128):
    img = misc.imread(image_path, mode='RGB')
    img = crop_center(img,178,178)
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def load_test_data_128(image_path, size=128):
    img = misc.imread(image_path, mode='RGB')
    #img = crop_center(img,178,178)
    #img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img
    
    
#def load_test_data(image_path, size=256):
#    #img = misc.imread(image_path, mode='RGB')
#    #img = cropimread(True,178,218,img)
#    #img = misc.imresize(img, [size, size])
#    img = np.asarray(pilimage_crop(image_path,178,178))
#    img = np.expand_dims(img, axis=0)
#    img = preprocessing(img)

#    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation1(image, aug_img_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_size, aug_img_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def augmentation(image, aug_img_size):
    seed = random.randint(0,2 **31 -1)
    ori_image_shape = tf.shape(image)
    print('pjw',image.get_shape())
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.central_crop(image,1)
    print('pjw',image.get_shape())
    image = tf.image.resize_images(image,[aug_img_size,aug_img_size])
    return image


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')
