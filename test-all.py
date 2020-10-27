import gc
from copy import deepcopy
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from modelUnetD import myUnet
import cv2
import gdal
import glob
import os
import data
import math
import matplotlib.image as mpimg
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Train_DIR = "E:\\Experiment data\\data\\Net3\\Train\\"
# Val_DIR = "E:\\Experiment data\\data\\Net3\\Val\\"
Test_DIR = "E:\\Experiment data\\cloud_R_256\\126035\\realtest\\"
Result_DIR = "E:\\Experiment data\\cloud_R_256\\126035\\realtest\\thmask\\"

if not os.path.exists(Result_DIR):
    os.makedirs(Result_DIR)

model = myUnet()
# model.load(r"G:\detection FCN PCAM\weight\channel attention\1000_weights_2019-05-25-16-10-34.h5")
model.load(r"H:\DenseMSFCN_CAM_cloud_detection\weight\202_weights_2019-09-18-01-31-55.h5")

imgsname = glob.glob(Test_DIR + "cloudy\\*.tif")
imgdatas = np.ndarray((1, 256, 256, 3), dtype=np.float32)

num = 0
patch_size = 256

# print(imgsname)

# def max_img(img):
#     x = np.zeros((patch_size, patch_size, 3))
#     # cloud = np.ndarray((patch_size, patch_size))
#     # shadow = np.ndarray((patch_size, patch_size))
#     # background = np.ndarray((patch_size, patch_size))
#     for i in range(0, patch_size):
#         for j in range(0, patch_size):
#             a = img[i][j]
#             b = np.argmax(a)
#             x[i][j][b] = 1
#     return x
#
#
# def threshold_img(img):
#     x = np.zeros((patch_size, patch_size, 3))
#     for i in range(0, patch_size):
#         for j in range(0, patch_size):
#             if (img[i][j][0] >= 0.4) and (img[i][j][0]>=img[i][j][1]):
#             # if (img[i][j][0] >= 0.5):
#                 x[i][j][0] = 1
#                 x[i][j][1] = 0
#                 x[i][j][2] = 0  #阴影
#
#             elif (img[i][j][1] >= 0.55) and (img[i][j][2]<img[i][j][1]):
#             # elif (img[i][j][1] >= 0.65):
#                 x[i][j][0] = 0
#                 x[i][j][1] = 1
#                 x[i][j][2] = 0 #cloud
#
#             else:
#                 x[i][j][0] = 0
#                 x[i][j][1] = 0
#                 x[i][j][2] = 1 #背景
#     return x
#
# def label_binary(labelimg):
#     # labelimg /= 255
#     labelimg[labelimg > 0.3] = 1
#     labelimg[labelimg <= 0.3] = 0
#     return labelimg
#
# def uint16to8(image):
#     array16 = np.array(image).astype('float32')
#     nmax = image.max()
#     nmin = image.min()
#     array8 = (array16 - nmin) * 255 / (nmax - nmin)
#
#     return array8
#
# for imgname in imgsname:
#     name = imgname[imgname.rindex("\\") + 1:]
#
#     img_dataset = gdal.Open(Test_DIR + "cloudy\\" + name)
#     rows = img_dataset.RasterYSize
#     cols = img_dataset.RasterXSize
#     couts = img_dataset.RasterCount
#     # img_array = img_dataset.ReadAsArray(0, 0, rows, cols)
#     array_data = np.zeros((rows, cols, couts))
#
#     for i in range(couts):
#         band = img_dataset.GetRasterBand(i + 1)
#         array_data[:, :, i] = band.ReadAsArray()
#
#     img = array_data
#     img = uint16to8(img)
#     # img = compress(img) #uint16->uint8
#     img = img.astype(np.float32)
#     # img = mpimg.imread(Test_DIR + "masked\\" + name)
#     # plt.imshow(img)
#     # plt.show()
#     # imgn = cv2.imread(Val_DIR + "masked\\" + name)
#     # plt.imshow(imgn)
#     # plt.show()
#     # img = img_to_array(img).astype('float32')
#     # img_dataset = gdal.Open(self.testdata + "Image\\" + name)
#     # im_width = img_dataset.RasterXSize
#     # im_height = img_dataset.RasterYSize
#     # im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
#     #         # print(im_data.shape)
#     # img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
#     #         # img, label = data_argmentation(img, label)
#     # img[:, :, 0] = im_data[0, :, :]
#     # img[:, :, 1] = im_data[1, :, :]
#     # img[:, :, 2] = im_data[2, :, :]
#     # img[:, :, 3] = im_data[3, :, :]
#     # img = img_to_array(img).astype('float32')
#     imgdatas[0] = img / 255
#     # mask = model.predict(imgdatas)
#     mask = model.predict(imgdatas)
#     # mask_fin = mask[3][0]
#     # mask_fin[mask_fin > 0.3] = 1
#     # mask_fin[mask_fin <= 0.3] = 0
#     (b, h, w, c) = mask.shape
#     label_normal = np.ndarray((b, h, w, c), dtype=np.uint8)
#     # mask = mask[0, :, :, :]
#     # mask[mask > 0.3] = 1
#     # mask[mask <= 0.3] = 0
#     label_normal[0] = max_img(mask[0, :, :, :])
#     result = label_normal[0] * 255
#     mask_img = array_to_img(result)
#
#     mask_img.save(Result_DIR + "\\%s" % name)
#
#     num = num + 1
#     if num % 100 == 0:
#         print("%d输出完毕！" % num)

def max_img(img):
    x = np.zeros((patch_size, patch_size, 3))
    # cloud = np.ndarray((patch_size, patch_size))
    # shadow = np.ndarray((patch_size, patch_size))
    # background = np.ndarray((patch_size, patch_size))
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            a = img[i][j]
            b = np.argmax(a)
            x[i][j][b] = 1
    return x


def threshold_img(img):
    x = np.zeros((patch_size, patch_size, 3))
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            if (img[i][j][0]>=0.01) and (img[i][j][0]>=img[i][j][1]):
                x[i][j][0] = 1
                x[i][j][1] = 0
                x[i][j][2] = 0
            elif (img[i][j][1]>=0.01) and (img[i][j][0]<img[i][j][1]):
                x[i][j][0] = 0
                x[i][j][1] = 1
                x[i][j][2] = 0
            else:
                x[i][j][0] = 0
                x[i][j][1] = 0
                x[i][j][2] = 1
    return x

for imgname in imgsname:
    name = imgname[imgname.rindex("\\") + 1:]
    img = mpimg.imread(Test_DIR + "cloudy\\" + name)
    # plt.imshow(img)
    # plt.show()
    # imgn = cv2.imread(Val_DIR + "masked\\" + name)
    # plt.imshow(imgn)
    # plt.show()
    img = img_to_array(img).astype('float32')

    # img_dataset = gdal.Open(self.testdata + "Image\\" + name)
    # im_width = img_dataset.RasterXSize
    # im_height = img_dataset.RasterYSize
    # im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
    #         # print(im_data.shape)
    # img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
    #         # img, label = data_argmentation(img, label)
    # img[:, :, 0] = im_data[0, :, :]
    # img[:, :, 1] = im_data[1, :, :]
    # img[:, :, 2] = im_data[2, :, :]
    # img[:, :, 3] = im_data[3, :, :]
    # img = img_to_array(img).astype('float32')

    imgdatas[0] = img / 255
    # mask = model.predict(imgdatas)
    mask = model.predict(imgdatas)

    # mask_fin = mask[3][0]
    # mask_fin[mask_fin > 0.3] = 1
    # mask_fin[mask_fin <= 0.3] = 0
    (b, h, w, c) = mask.shape
    label_normal = np.ndarray((b, h, w, c), dtype=np.uint8)

    # mask = mask[0, :, :, :]
    # mask[mask > 0.3] = 1
    # mask[mask <= 0.3] = 0
    # label_normal[0] = threshold_img(mask[0, :, :, :])
    label_normal[0] = threshold_img(mask[0, :, :, :])

    result = label_normal[0] * 255


    mask_img = array_to_img(result)
    mask_img.save(Result_DIR + "\\%s" % name)

    num = num + 1
    if num % 100 == 0:
        print("%d输出完毕！" % num)


