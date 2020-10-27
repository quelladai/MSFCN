from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
from random import shuffle
import cv2
from PIL import Image
import matplotlib.image as mpimg
# import gdal
# from data_argumentation import *
from keras.utils.np_utils import to_categorical
from pylab import *
# from imgaug import augmenters as iaa
# import imgaug as ia
import numpy as np
from scipy import misc
import os
import gdal


def uint16to8(image):
    array16 = np.array(image).astype('float32')
    nmax = image.max()
    nmin = image.min()
    array8 = (array16 - nmin) * 255 / (nmax - nmin)

    return array8


def createdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def label_binary(labelimg):
    # labelimg /= 255
    labelimg[labelimg > 0.3] = 1
    labelimg[labelimg <= 0.3] = 0
    return labelimg

# Prediction_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Train\\"
# masked_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Val\\"
#
# imgs = glob.glob(masked_DIR + "masked\\*.tif")
# random.shuffle(imgs)
# cnt = 0
# while 1:
#     for imgname in imgs:
#         midname = imgname[imgname.rindex("\\") + 1:]
#         masked = cv2.imread(masked_DIR + "masked\\" + midname)
#         prediction = cv2.imread(Prediction_DIR + "prediction\\" + midname)
#
#         masked = img_to_array(masked).astype('float32')
#         prediction = img_to_array(prediction).astype('float32')
#
#         cha = masked - prediction
#
#         label = img_to_array(label).astype('float32')
#         img /= 255
#         label = label_binary(label)
#         imgdatas.append(img)
#         imglabels.append(label)
#         cnt += 1

def generatedata_0riginal(path,batchsize):
    imgs = glob.glob(path + "masked\\*.tif")
    imgdatas = []
    imglabels = []
    class_number = 5
    cnt = 0
    # num = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            # img = cv2.imread(path + "masked\\" + midname)
            img = mpimg.imread(path + "masked\\" + midname)
            # label = cv2.imread(path + "single-mask\\" + midname, cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(path + "overall-mask\\" + midname,  cv2.IMREAD_COLOR)
            label = mpimg.imread(path + "overall-mask\\" + midname)
            # label = cv2.imread(path + "mask\\118032\\" + midname)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            label /= 255
            # label = label_binary(label)
            # label[label == 255] = 0
            # newlabel = to_categorical(label, class_number)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []
                # num += 2
                # print(num)


def generatedata_0riginal_1(path,batchsize):
    imgs = glob.glob(path + "masked\\*.tif")
    shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(path + "masked\\" + midname)
            label = cv2.imread(path + "overall-mask\\" + midname)
            # label = cv2.imread(path + "mask\\118032\\" + midname)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []

def generatedata1(path, batchsize):
    imgs = glob.glob(path + "image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = mpimg.imread(path + "image\\" + midname)
            label = mpimg.imread(path + "label\\" + midname, cv2.IMREAD_GRAYSCALE)
            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            img /= 255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []

def generatedata(path, batchsize):
    imgs = glob.glob(path + "cloudy\\*.tif")
    shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            # img = mpimg.imread(path + "cloudy\\" + midname)
            # img = cv2.imread(path + "masked\\" + midname, cv2.IMREAD_COLOR)
            # img = tif.imread(path + "masked\\" + midname)
            img_dataset = gdal.Open(path + "cloudy\\" + midname)
            rows = img_dataset.RasterYSize
            cols = img_dataset.RasterXSize
            couts = img_dataset.RasterCount
            # img_array = img_dataset.ReadAsArray(0, 0, rows, cols)
            array_data = np.zeros((rows, cols, couts))

            for i in range(couts):
                band = img_dataset.GetRasterBand(i + 1)
                array_data[:, :, i] = band.ReadAsArray()

            img = array_data
            img = uint16to8(img)
            # img = compress(img) #uint16->uint8
            img = img.astype(np.float32)
            # img = cv2.imread(path + "masked\\" + midname, -1)
            # label = mpimg.imread(path + "cloud\\" + midname, cv2.IMREAD_GRAYSCALE)
            label = mpimg.imread(path + "overall\\" + midname)

            # img, label = data_argmentation(img, label)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')

            img = img / 255
            label = label/255
            label = label_binary(label)
            imgdatas.append(img)
            imglabels.append(label)
            cnt += 1
            if cnt == batchsize:
                imgdatas = np.asarray(imgdatas)
                labeldatas = np.asarray(imglabels)
                yield (imgdatas, labeldatas)
                cnt = 0
                imgdatas = []
                imglabels = []


def getnumpyname(dir):
    listname = []
    for filename in os.listdir(dir):
        if os.path.splitext(filename)[1] == '.npy':
            filename = os.path.splitext(filename)[0]
            listname.append(filename)
    return listname


def apply_aug(image):
    images = []

    flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image
    images.append(flipper.augment_image(image))  # horizontally flip image 0

    vflipper = iaa.Flipud(1)  # vertically flip each input image with 90% probability
    images.append(vflipper.augment_image(image))  # probably vertically flip image 1
    # blurer = iaa.GaussianBlur(sigma=(0, 3.0))
    # images.append(blurer.augment_image(image))  # blur image 2 by a sigma of 3.0
    # crop_and_pad = iaa.CropAndPad(px=(0, 30))  # crop images from each side by 0 to 16px (randomly chosen)
    # images.append(crop_and_pad.augment_image(image))
    # inverter = iaa.Invert(0.05)
    # images.append(inverter.augment_image(image))

    # contrast_normalization = iaa.ContrastNormalization((0.5, 2.0))
    # images.append(contrast_normalization.augment_image(image))
    #
    # add_process = iaa.Add((-10, 10), per_channel=0.5)
    # images.append(add_process.augment_image(image))
    #
    # sharpen_process = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
    # images.append(sharpen_process.augment_image(image))
    #
    # emboss_process = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # emboss images
    # images.append(emboss_process.augment_image(image))
    rot_process_1 = iaa.Rot90(1)
    images.append(rot_process_1.augment_image(image))

    rot_process_2 = iaa.Rot90(2)
    images.append(rot_process_2.augment_image(image))

    # rot_process_3 = iaa.Rot90(3)
    # images.append(rot_process_3.augment_image(image))

    # elastic_transformation_process = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
    # images.append(elastic_transformation_process.augment_image(image))
    #
    # perspectivetransform_process = iaa.PerspectiveTransform(scale=(0.01, 0.1))
    # images.append(perspectivetransform_process.augment_image(image))
    #
    # averageblur_process = iaa.AverageBlur(k=(2, 7))
    # images.append(averageblur_process.augment_image(image))
    #
    # medianblur_process = iaa.MedianBlur(k=(3, 11))
    # images.append(medianblur_process.augment_image(image))

    return np.array(images, dtype=np.uint8)

def numpygeneratedata4(path, batchsize):
    imgs = getnumpyname(path + "img\\")
    # imgdatas = []
    # imglabels = []
    # imglabels_sub2 = []
    # imglabels_sub4 = []
    # imglabels_sub8 = []

    class_number = 16
    cnt = 0
    while 1:
        for midname in imgs:
            img = np.load(path + "img\\" + midname + '.npy')
            img = np.array(img, dtype=float).astype('float32')
            img /= 255
            label = cv2.imread(path + "label\\" + midname + '.tif', cv2.IMREAD_GRAYSCALE)
            label[label == 255] = 15
            label = np.array(label).astype('float32')
            label_sub2 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
            label_sub4 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
            label_sub8 = cv2.resize(label, (32, 32), interpolation=cv2.INTER_AREA)
            # label = img_to_array(label).astype('float32')
            newlabel = to_categorical(label, class_number)
            newlabel_sub2 = to_categorical(label_sub2, class_number)
            newlabel_sub4 = to_categorical(label_sub4, class_number)
            newlabel_sub8 = to_categorical(label_sub8, class_number)

            # imgdatas.append(img)
            #
            # imglabels.append(newlabel)
            # imglabels_sub2.append(newlabel_sub2)
            # imglabels_sub4.append(newlabel_sub4)
            # imglabels_sub8.append(newlabel_sub8)

            imgdatas = apply_aug(img)
            imglabels = apply_aug(newlabel)
            imglabels_sub2 = apply_aug(newlabel_sub2)
            imglabels_sub4 = apply_aug(newlabel_sub4)
            imglabels_sub8 = apply_aug(newlabel_sub8)

            cnt += 1
            if cnt == batchsize:
                yield (
                np.asarray(imgdatas), [np.asarray(imglabels), np.asarray(imglabels), np.asarray(imglabels_sub2),
                                     np.asarray(imglabels_sub4), np.asarray(imglabels_sub8)])
                cnt = 0
                # imgdatas = []
                # imglabels = []
                # imglabels_sub2 = []
                # imglabels_sub4 = []
                # imglabels_sub8 = []

# def apply_aug(image):
#     images = []
#
#     flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image
#     images.append(flipper.augment_image(image))  # horizontally flip image 0
#
#     vflipper = iaa.Flipud(1)  # vertically flip each input image with 90% probability
#     images.append(vflipper.augment_image(image))  # probably vertically flip image 1
#     # blurer = iaa.GaussianBlur(sigma=(0, 3.0))
#     # images.append(blurer.augment_image(image))  # blur image 2 by a sigma of 3.0
#     crop_and_pad = iaa.CropAndPad(px=(0, 30))  # crop images from each side by 0 to 16px (randomly chosen)
#     images.append(crop_and_pad.augment_image(image))
#     # inverter = iaa.Invert(0.05)
#     # images.append(inverter.augment_image(image))
#
#     # contrast_normalization = iaa.ContrastNormalization((0.5, 2.0))
#     # images.append(contrast_normalization.augment_image(image))
#     #
#     # add_process = iaa.Add((-10, 10), per_channel=0.5)
#     # images.append(add_process.augment_image(image))
#     #
#     # sharpen_process = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
#     # images.append(sharpen_process.augment_image(image))
#     #
#     # emboss_process = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))  # emboss images
#     # images.append(emboss_process.augment_image(image))
#     rot_process_1 = iaa.Rot90(1)
#     images.append(rot_process_1.augment_image(image))
#
#     rot_process_2 = iaa.Rot90(2)
#     images.append(rot_process_2.augment_image(image))
#
#     rot_process_3 = iaa.Rot90(3)
#     images.append(rot_process_3.augment_image(image))
#
#     # elastic_transformation_process = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
#     # images.append(elastic_transformation_process.augment_image(image))
#     #
#     # perspectivetransform_process = iaa.PerspectiveTransform(scale=(0.01, 0.1))
#     # images.append(perspectivetransform_process.augment_image(image))
#     #
#     # averageblur_process = iaa.AverageBlur(k=(2, 7))
#     # images.append(averageblur_process.augment_image(image))
#     #
#     # medianblur_process = iaa.MedianBlur(k=(3, 11))
#     # images.append(medianblur_process.augment_image(image))
#
#     return np.array(images, dtype=np.uint8)

def numpygeneratedata(path, batchsize):
    imgs = getnumpyname(path + "img\\")

    class_number = 5
    cnt = 0
    while 1:
        for midname in imgs:
            img = cv2.imread(path + "img\\" + midname + '.tif')
            img = np.array(img, dtype=float).astype('float32')
            img /= 255
            label = cv2.imread(path + "label\\" + midname + '.tif', cv2.IMREAD_GRAYSCALE)
            label[label == 255] = 0
            label = np.array(label).astype('float32')

            newlabel = to_categorical(label, class_number)

            imgdatas = apply_aug(img)
            imglabels = apply_aug(newlabel)

            cnt += 1
            if cnt == batchsize:
                yield (
                np.asarray(imgdatas), np.asarray(imglabels))
                cnt = 0

# def numpygeneratedata(path, batchsize):
#     imgs = getnumpyname(path + "img\\")
#     imgdatas = []
#     imglabels = []
#
#     class_number = 5
#     cnt = 0
#     while 1:
#         for midname in imgs:
#             img = cv2.imread(path + "img\\" + midname + '.tif')
#             # label = img_to_array(label).astype('float32')
#             # img = np.load(path + "img\\" + midname + '.npy')
#             img = np.array(img, dtype=float).astype('float32')
#             img /= 255
#             # img = apply_aug(img)
#             label = cv2.imread(path + "label\\" + midname + '.tif', cv2.IMREAD_GRAYSCALE)
#             label[label == 255] = 0
#             # label = img_to_array(label).astype('float32')
#             label = np.array(label).astype('float32')
#             # label = apply_aug(label)
#             newlabel = to_categorical(label, class_number)
#             # label = label_binary(label)
#             imgdatas.append(img)
#             imglabels.append(newlabel)
#
#             cnt += 1
#             if cnt == batchsize:
#                 imgdatas = np.asarray(imgdatas)
#                 labeldatas = np.asarray(imglabels)
#                 yield (imgdatas, labeldatas)
#                 cnt = 0
#                 imgdatas = []
#                 imglabels = []

    # imgs = glob.glob(path + "img\\*.npy")
    # # random.shuffle(imgs)
    # imgdatas = []
    # imglabels = []
    # cnt = 0
    # while 1:
    #     for imgname in imgs:
    #         midname = imgname[imgname.rindex("\\") + 1:]   #定位到最后一个\\
    #         img = np.load(path + "img\\" + midname)
    #         img = np.array(img, dtype=float).astype('float32')
    #         img /= 255
    #         label = cv2.imread(path + "label\\" + midname, cv2.IMREAD_GRAYSCALE)
    #         # img, label = data_argmentation(img, label)
    #         # label = img_to_array(label).astype('float32')
    #         label = array(label)
    #         print(label.shape)
    #
    #         label = to_categorical(label, 16)
    #         imgdatas.append(img)
    #         imglabels.append(label)
    #         cnt += 1
    #         if cnt == batchsize:
    #             imgdatas = np.asarray(imgdatas)
    #             labeldatas = np.asarray(imglabels)
    #             yield (imgdatas, labeldatas)
    #             cnt = 0
    #             imgdatas = []
    #             imglabels = []

def generatedata4(path, batchsize):
    imgs = glob.glob(path + "masked\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    imglabels_sub2 = []
    imglabels_sub4 = []
    imglabels_sub8 = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            img = cv2.imread(path + "masked\\" + midname)
            label = cv2.imread(path + "mask\\" + midname, cv2.IMREAD_GRAYSCALE)
            label_sub2 = cv2.resize(label, (256, 256), interpolation=cv2.INTER_AREA)
            label_sub4 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
            label_sub8 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
            img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            label_sub2 = img_to_array(label_sub2).astype('float32')
            label_sub4 = img_to_array(label_sub4).astype('float32')
            label_sub8 = img_to_array(label_sub8).astype('float32')
            img /= 255
            label = label_binary(label)
            label_sub2 = label_binary(label_sub2)
            label_sub4 = label_binary(label_sub4)
            label_sub8 = label_binary(label_sub8)
            imgdatas.append(img)
            imglabels.append(label)
            imglabels_sub2.append(label_sub2)
            imglabels_sub4.append(label_sub4)
            imglabels_sub8.append(label_sub8)
            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), [np.array(imglabels_sub8), np.array(imglabels_sub4), np.array(imglabels_sub2),
                                            np.array(imglabels), np.array(imglabels)])
                cnt = 0
                imgdatas = []
                imglabels = []
                imglabels_sub2 = []
                imglabels_sub4 = []
                imglabels_sub8 = []

def generatedata_multichannel(path, batchsize):
    imgs = glob.glob(path + "Image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            #img = cv2.imread(path + "Image\\" + midname)
            label = cv2.imread(path + "Label\\" + midname, cv2.IMREAD_GRAYSCALE)
            label = img_to_array(label).astype('float32')
            label = label_binary(label)
            imglabels.append(label)

            img_dataset = gdal.Open(path + "Image\\" + midname)
            im_width = img_dataset.RasterXSize
            im_height = img_dataset.RasterYSize
            im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
            #print(im_data.shape)
            img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
            # img, label = data_argmentation(img, label)
            img[:, :, 0] = im_data[0, :, :]
            img[:, :, 1] = im_data[1, :, :]
            img[:, :, 2] = im_data[2, :, :]
            img[:, :, 3] = im_data[3, :, :]
            #img = img_to_array(img).astype('float32')
            #mg /= (255*255)
            imgdatas.append(img)

            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), np.array(imglabels))
                cnt = 0
                imgdatas = []
                imglabels = []


def generatedata4_multichannel(path, batchsize):
    imgs = glob.glob(path + "Image\\*.tif")
    random.shuffle(imgs)
    imgdatas = []
    imglabels = []
    imglabels_sub2 = []
    imglabels_sub4 = []
    imglabels_sub8 = []
    cnt = 0
    while 1:
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\") + 1:]
            #img = cv2.imread(path + "Image\\" + midname)
            img_dataset = gdal.Open(path + "Image\\" + midname)
            im_width = img_dataset.RasterXSize
            im_height = img_dataset.RasterYSize
            im_data = img_dataset.ReadAsArray(0, 0, im_width, im_height)
            # print(im_data.shape)
            img = np.ndarray((im_width, im_height, 4), dtype=np.float32)
            # img, label = data_argmentation(img, label)
            img[:, :, 0] = im_data[0, :, :]
            img[:, :, 1] = im_data[1, :, :]
            img[:, :, 2] = im_data[2, :, :]
            img[:, :, 3] = im_data[3, :, :]

            label = cv2.imread(path + "Label-cloud\\" + midname, cv2.IMREAD_GRAYSCALE)
            label_sub2 = cv2.resize(label, (256, 256), interpolation=cv2.INTER_AREA)
            label_sub4 = cv2.resize(label, (128, 128), interpolation=cv2.INTER_AREA)
            label_sub8 = cv2.resize(label, (64, 64), interpolation=cv2.INTER_AREA)
            #img = img_to_array(img).astype('float32')
            label = img_to_array(label).astype('float32')
            label_sub2 = img_to_array(label_sub2).astype('float32')
            label_sub4 = img_to_array(label_sub4).astype('float32')
            label_sub8 = img_to_array(label_sub8).astype('float32')
            img /= (255*255)
            label = label_binary(label)
            label_sub2 = label_binary(label_sub2)
            label_sub4 = label_binary(label_sub4)
            label_sub8 = label_binary(label_sub8)
            imgdatas.append(img)
            imglabels.append(label)
            imglabels_sub2.append(label_sub2)
            imglabels_sub4.append(label_sub4)
            imglabels_sub8.append(label_sub8)
            cnt += 1
            if cnt == batchsize:
                yield (np.array(imgdatas), [np.array(imglabels_sub8), np.array(imglabels_sub4), np.array(imglabels_sub2),
                                            np.array(imglabels), np.array(imglabels)])
                cnt = 0
                imgdatas = []
                imglabels = []
                imglabels_sub2 = []
                imglabels_sub4 = []
                imglabels_sub8 = []

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    path = "F:/HED-BSDS"
    f = open("F:/HED-BSDS/train_pair.txt", "r")
    files = f.readlines()
    name = files[0].split(" ")
    img = cv2.imread(os.path.join(path, name[0]))
    label = cv2.imread(os.path.join(path, name[1][:-1]))
    img = cv2.resize(img, (512, 512))
    label = cv2.resize(label, (512, 512))
    cv2.imshow("1", img)
    cv2.imshow("2", label)
    cv2.waitKey()