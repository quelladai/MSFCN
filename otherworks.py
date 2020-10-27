from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob
import cv2
import shutil
import PIL.Image as Image
import math

def transform():
    path = "F:\\allimage\\"
    imgsname = glob.glob(path + "label\\*.jpg")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:-4]
        img = load_img(path + "label\\" + name + ".jpg")
        img.save(path + "tif\\label\\" + name + ".tif")


def clip():
    path = "G:\\zhejiang\\wp\\image\\"
    cc_path = "G:\\zhejiang\\wp\\clip\\image\\"
    imgsname = glob.glob(path + "*.tif")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:-4]
        img = cv2.imread(imgname)
        h, w, _ = img.shape
        nw = math.ceil(w / 512)
        nh = math.ceil(h / 512)
        newimg = np.zeros((nh * 512, nw * 512, 3), img.dtype)
        newimg[0:h, 0:w] = img
        n = 1
        for i in range(nh):
            for j in range(nw):
                img1 = newimg[i * 512:i * 512 + 512, j * 512:j * 512 + 512]
                cv2.imwrite(cc_path + name + "_" + str(n) + ".tif", img1)
                n = n + 1
        print(name)


def cliplabel():
    path = "G:\\zhejiang\\wp\\label\\"
    cc_path = "G:\\zhejiang\\wp\\clip\\label\\"
    imgsname = glob.glob(path + "*.tif")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:-4]
        img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
        img[img == 0] = 1
        img[img == 255] = 0
        img[img == 1] = 255
        h, w = img.shape
        nw = math.ceil(w / 512)
        nh = math.ceil(h / 512)
        newimg = np.zeros((nh * 512, nw * 512), img.dtype)
        newimg[0:h, 0:w] = img
        n = 1
        for i in range(nh):
            for j in range(nw):
                img1 = newimg[i * 512:i * 512 + 512, j * 512:j * 512 + 512]
                cv2.imwrite(cc_path + name + "_" + str(n) + ".tif", img1)
                n = n + 1
        print(name)

def clip2():
    path = "F:\\allimage\\2\\label\\"
    imgsname = glob.glob(path + "*.tif")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        label = load_img(path + name)
        n = 0
        for i in range(2):
            for j in range(2):
                label1 = label.crop(((j*512), (i*512), (j*512+512), (i*512+512)))
                label1.save(path + "label\\" + name[:-4] + "_" + str(n) + ".tif")
                n = n + 1
        n = 0

def w_to_b():
    path = "F:\\allimage\\2\\label\\transform\\"
    imgsname = glob.glob(path + "*.tif")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(path + name)
        img = img_to_array(img).astype(np.float32)
        img[img >= 50] = 50
        img[img < 50] = 255
        img[img == 50] = 0
        img = array_to_img(img)
        img.save(path + "1\\" + name)


def extract():
    path = "F:\\ISPRS\\"
    imgs = glob.glob(path + "5_Labels_for_participants\\*.tif")
    for imgname in imgs:
        name = imgname[imgname.rindex("\\") + 1:]
        image = cv2.imread(path + "5_Labels_for_participants\\" + name, cv2.IMREAD_GRAYSCALE)
        print(image.shape)
        image_arr = img_to_array(image)
        image_arr[image_arr < 40] = 0
        image_arr[image_arr > 0] = 255
        image_arr[image_arr == 0] = 100
        image_arr[image_arr == 255] = 0
        image_arr[image_arr == 100] = 255
        mask = array_to_img(image_arr)
        mask.save(path + "build\\" + name)


def search():
    f = open("E:\\ArcGIS\\building\\name.txt", "r")
    names = f.readlines()
    for name in names:
        shutil.copy("F:\\allimage\\tif\\image\\512\\" + name[:-1], "F:\\allimage\\good\\image\\")


def to_jpg():
    path = "F:\\ISPRS\\"
    imgsname = glob.glob(path + "2_Ortho_RGB\\*.tif")
    # print(imgsname)
    for imgname in imgsname:
        # img = load_img(imgname)
        img = Image.open(imgname)
        name = imgname[imgname.rindex("\\") + 1:-8]
        # print(path , "jpg\\image\\" , name ,".jpg")
        img.save(path + "jpg\\image\\" + name + ".tif")


def R_to_b():
    path = "F:\\Mnih\\512\\all\\label\\"
    imgsname = glob.glob(path + "*.tif")
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(path + name, grayscale=True)
        img = img_to_array(img)
        img[img > 10] = 255
        img[img <= 10] = 0
        img = array_to_img(img)
        img.save(path + "t\\" + name)


def pinjie():
    files = glob.glob("F:\\python_progect\\unet\\results\\*.tif")
    # files = glob.glob("D:\\cx\\test\\label\\*.tif")
    base_img = Image.open(files[0])
    base_size = base_img.size
    new_img = Image.new('L', (base_size[0]*56+113, base_size[1]*59+312), 0) # w,h
    x = y = 0
    cont = 0
    for h_num in range(59):
        for w_num in range(56):
            img = Image.open("F:\\python_progect\\unet\\results\\" + str(cont) + ".tif")
            cont += 1
            new_img.paste(img, (x, y))
            x += base_size[0]
        x = 0
        y += base_size[0]
        print(cont)
    # new_img.show()
    new_img.save("G:\\zhejiang\\wp\\wp_label.tif")


def to_label_img():
    path = "F:\\python_progect\\unet\\results\\"
    imgsname = glob.glob(path + "*.tif")
    n = 0
    for imgname in imgsname:
        name = imgname[imgname.rindex("\\") + 1:]
        img = load_img(imgname)
        img = img_to_array(img).astype(np.float32)
        img[img >= 255*0.5] = 255
        img[img < 255*0.5] = 0
        img = array_to_img(img)
        img.save(path + name)
        n = n + 1
        if n % 1000 == 0:
            print(n)


def generate_bsd(path):
    imgs = glob.glob(path + "image\\*.jpg")
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = cv2.imread(path + "image\\" + midname)
        label = cv2.imread(path + "label\\" + midname[:-4] + ".tif", cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if shape[0] > shape[1]:
            img = np.rot90(img)
            label = np.rot90(label)
        img = img[0:320, 0:480]
        label = label[0:320, 0:480]
        cv2.imwrite(path + "image_T\\" + midname[:-4] + ".tif", img)
        cv2.imwrite(path + "label_T\\" + midname[:-4] + ".tif", label)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    f = open("F:/HED-BSDS/train_pair.txt", "r")
    lines = f.readlines()
    print(len(lines))