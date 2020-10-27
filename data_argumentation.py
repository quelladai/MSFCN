import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def draw_image(data_argumentation, numpy_array,imgname, result_images):
    temporary_directory = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Train\\directory"
    # os.mkdir(temporary_directory)

    # generate 9 images by ImageDataGenerator
    # generate_data = data_argumentation.flow(numpy_array, batch_size=1, save_to_dir=temporary_directory, save_prefix=imgname, save_format='tif')
    generate_data = data_argumentation.flow(numpy_array, batch_size=1)

    for i in range(9):
        batch = generate_data.next()
    #
    images = glob.glob(os.path.join(temporary_directory, "*.tif"))
    figure = plt.figure()
    gridspec_image = gridspec.GridSpec(3, 3)
    gridspec_image.update(wspace=0.1, hspace=0.1)
    #
    for i in range(9):
        image = load_img(images[i])
        plt.subplot(gridspec_image[i])
        plt.imshow(image, aspect='auto')
        plt.axis("off")
    plt.savefig(r'E:\Experiment data\data-for-detection\lansat-fintune\Train\directory\rotation_{}_{}.png')
    plt.close("all")

    # shutil.rmtree(temporary_directory)


if __name__ == '__main__':
    # preprocessing image
    IMAGE_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Test\\"
    imgs = glob.glob(IMAGE_DIR + "masked\\*.tif")
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = cv2.imread(IMAGE_DIR + "masked\\" + midname)
        # label = cv2.imread(path + "single-mask\\" + midname, cv2.IMREAD_GRAYSCALE)
        numpy_array = img_to_array(img).astype('float32')
        numpy_array = np.expand_dims(numpy_array, axis=0)

        data_argumentation = ImageDataGenerator(rotation_range=90)
        draw_image(data_argumentation, numpy_array, imgname, "result_rotation.jpg")


    # IMAGE_DIR = "E:\\Experiment data\\data-for-detection\\lansat-fintune\\Test\\masked\\118032_39.tif"
    # image = load_img(IMAGE_DIR)
    # numpy_array = img_to_array(image)
    # numpy_array = np.expand_dims(numpy_array, axis=0)
    # # load ImageDataGenerator & processing images
    # data_argumentation = ImageDataGenerator(rotation_range=90)
    # draw_image(data_argumentation, numpy_array, "result_rotation.jpg")
    # imgs = glob.glob(IMAGE_DIR + "masked\\*.tif")
    # while 1:
    #     for imgname in imgs:
    #         midname = imgname[imgname.rindex("\\") + 1:]
    #         img = cv2.imread(IMAGE_DIR + "masked\\" + midname)
    #         # label = cv2.imread(path + "single-mask\\" + midname, cv2.IMREAD_GRAYSCALE)
    #         numpy_array = img_to_array(img).astype('float32')
    #         numpy_array = np.expand_dims(numpy_array, axis=0)
    #
    #         data_argumentation = ImageDataGenerator(rotation_range=90)
    #         draw_image(data_argumentation, numpy_array, "result_rotation.jpg")

            # data_argumentation = ImageDataGenerator(width_shift_range=0.2)
            # draw_image(data_argumentation, numpy_array, "result_width_shift.jpg")
            #
            # data_argumentation = ImageDataGenerator(height_shift_range=0.2)
            # draw_image(data_argumentation, numpy_array, "result_height_shift.jpg")
            # shutil.move("result_height_shift.jpg", "Data_Argument")

            # data_argumentation = ImageDataGenerator(shear_range=0.78)
            # draw_image(data_argumentation, numpy_array, "result_shear.jpg")
            # shutil.move("result_shear.jpg", "Data_Argument")
            # data_argumentation = ImageDataGenerator(zoom_range=0.5)
            # draw_image(data_argumentation, numpy_array, "result_zoom.jpg")
            # shutil.move("result_zoom.jpg", "Data_Argument")
            # data_argumentation = ImageDataGenerator(channel_shift_range=100)
            # draw_image(data_argumentation, numpy_array, "result_channel_shift.jpg")
            # shutil.move("result_channel_shift.jpg", "Data_Argument")
            # data_argumentation = ImageDataGenerator(horizontal_flip=True)
            # draw_image(data_argumentation, numpy_array, "result_horizontal_flip.jpg")
            # shutil.move("result_horizontal_flip.jpg", "Data_Argument")
            # data_argumentation = ImageDataGenerator(vertical_flip=True)
            # draw_image(data_argumentation, numpy_array, "result_vertical_flip.jpg")
            # shutil.move("result_vertical_flip.jpg", "Data_Argument")
            # data_argumentation = ImageDataGenerator(samplewise_center=True)
            # draw_image(data_argumentation, numpy_array, "result_samplewise_center.jpg")
            # shutil.move("result_samplewise_center.jpg", "Data_Argument")



