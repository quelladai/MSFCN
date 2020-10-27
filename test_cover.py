# import torch
import keras as K
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# from torch.autograd import Variable
import numpy as np
import cv2 as cv
# from dataInput import LoadTest, LoadTestFullImage
from glob import glob
# from models.DenseASPP import denseASPP121
# from models.net import DFAVGG
from collections import Counter
import time
import gc
from copy import deepcopy
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from modelUnetD import myUnet
import data
import math
import matplotlib.image as mpimg



def label2RGB(label):
    '''
    居民地 1 红色(255,0,0)
    道路 2 绿色(0,255,0)
    水体 3 蓝色(0,0,255)
    植被 4 黄色(255,255,0)
    其它类 0 黑色(0,0,0)
    '''
    m, n = label.shape
    rgb = np.zeros((m, n, 3), dtype=np.uint8)
    rgb[label == 1] = np.array((0, 0, 255), dtype=np.uint8)
    rgb[label == 2] = np.array((0, 255, 0), dtype=np.uint8)
    rgb[label == 3] = np.array((255, 0, 0), dtype=np.uint8)
    rgb[label == 4] = np.array((0, 255, 255), dtype=np.uint8)
    return rgb

def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)

def metrics(confu_mat_total, save_path, backgound=False):
    '''
    :param confu_mat: 总的混淆矩阵
    backgound：是否干掉背景
    :return: excel写出混淆矩阵, precision，recall，IOU，f-score
    FinalClass,False表示去掉最后一个类别，计算mIou, mf-score
    '''
    class_num = confu_mat_total.shape[0]

    if backgound:
        confu_mat_total = confu_mat_total[1:class_num, 1:class_num]  # 干掉背景
        class_num = confu_mat_total.shape[0]

    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)

    # 将混淆矩阵写入excel中
    TP = []  # 识别中每类分类正确的个数

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出precision，recall, f1-score，f1-m以及mIOU

    f1_m = []
    iou_m = []
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    with open(save_path + 'accuracy.txt', 'w') as f:
        f.write('OA:\t%.4f\n' % (oa*100))
        f.write('kappa:\t%.4f\n' % (kappa*100))
        f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m)*100))
        f.write('mIou:\t%.4f\n' % (np.mean(iou_m)*100))

        # 写出precision
        f.write('precision:\n')
        for i in range(class_num):
            f.write('%.4f\t' % (float(TP[i]/raw_sum[i])*100))
        f.write('\n')

        # 写出recall
        f.write('recall:\n')
        for i in range(class_num):
            f.write('%.4f\t' % (float(TP[i] / col_sum[i])*100))
        f.write('\n')

        # 写出f1-score
        f.write('f1-score:\n')
        for i in range(class_num):
            f.write('%.4f\t' % (float(f1_m[i])*100))
        f.write('\n')

        # 写出 IOU
        f.write('Iou:\n')
        for i in range(class_num):
            f.write('%.4f\t' % (float(iou_m[i])*100))
        f.write('\n')

def max_img(img):
    x = np.zeros((patch_size, patch_size, 3))
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
            if (img[i][j][0] >= 0.5) and (img[i][j][0]>=img[i][j][1]):
                x[i][j][0] = 1
                x[i][j][1] = 0
                x[i][j][2] = 0
            elif (img[i][j][1] >= 0.5) and (img[i][j][0]<img[i][j][1]):
                x[i][j][0] = 0
                x[i][j][1] = 1
                x[i][j][2] = 0
            else:
                x[i][j][0] = 0
                x[i][j][1] = 0
                x[i][j][2] = 1
    return x

def label_binary(labelimg):
    # labelimg /= 255
    labelimg[labelimg > 0.3] = 1
    labelimg[labelimg <= 0.3] = 0
    return labelimg

class TestFullImage(object):
    def __init__(self, img_path, model, save_path, class_num, patch_size=320, overlap_rate=1/8, label_path=''):
        self.img_path = img_path  # 预测原始影像
        self.model = model
        self.save_path = save_path
        self.class_num = class_num
        self.patch_size = patch_size  # 预测影像大小，影像大小能被重叠率的倒数整除
        self.overlap_rate = overlap_rate  # 设置预测影像重叠率
        self.label_path = label_path  # 真值位置

    def predict_image_overlap_rate(self, model, img_path):
        # subsidiary value for the prediction of an image with overlap
        boder_value = int(self.patch_size * self.overlap_rate / 2)
        double_bv = boder_value*2
        stride_value = self.patch_size - double_bv
        most_value = stride_value + boder_value

        # an image for prediction
        basename = os.path.basename(img_path)
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        m, n, _ = img.shape
        tmp = (m - double_bv) // stride_value  # 剔除重叠部分相当于无缝裁剪
        new_m = tmp if (m - double_bv) % stride_value == 0 else tmp + 1
        tmp = (n - double_bv) // stride_value
        new_n = tmp if (n - double_bv) % stride_value == 0 else tmp + 1
        new_full_img = np.zeros((new_m*stride_value+double_bv, new_n*stride_value+double_bv, 3), dtype=np.uint8)
        new_full_img[:m, :n, :] = img

        FullPredict = np.zeros((new_m*stride_value+double_bv, new_n*stride_value+double_bv), dtype=np.uint8)


        for i in range(new_m):
            for j in range(new_n):
                tmp_img = new_full_img[
                          i*stride_value:((i+1)*stride_value+double_bv),
                          j*stride_value:((j+1)*stride_value+double_bv), :]
                tmp_img = img_to_array(tmp_img).astype('float32')
                tmp_img = tmp_img / 255
                # print(tmp_img.shape)
                # tmp_img = load_data(tmp_img)
                # tmp_img = Variable(tmp_img)
                # tmp_img = tmp_img.to(DEVICE).unsqueeze(0)
                tmp_img = np.expand_dims(tmp_img, axis=0)
                # print(tmp_img.shape)
                output = model.predict(tmp_img)[0]
                # print(output.shape)
                # pred = output.max(1)[1].data.cpu().numpy().squeeze(0)  # [0]
                pred = np.argmax(output, axis=-1)

                if i == 0 and j == 0:  # 左上角
                    FullPredict[0:most_value, 0:most_value] = pred[0:most_value, 0:most_value]
                elif i == 0 and j == new_n-1:  # 右上角
                    FullPredict[0:most_value, -most_value:] = pred[0:most_value, boder_value:]
                elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                    FullPredict[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[0:most_value, boder_value:most_value]

                elif i == new_m - 1 and j == 0:  # 左下角
                    FullPredict[-most_value:, 0:most_value] = pred[boder_value:, :-boder_value]
                elif i == new_m - 1 and j == new_n - 1:  # 右下角
                    FullPredict[-most_value:, -most_value:] = pred[boder_value:, boder_value:]
                elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                    FullPredict[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:, boder_value:-boder_value]

                elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                        pred[boder_value:-boder_value, 0:-boder_value]
                elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                        pred[boder_value:-boder_value, boder_value:]
                else:  # 中间情况
                    FullPredict[
                    boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                    boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:-boder_value, boder_value:-boder_value]

        predict = FullPredict[0:m, 0:n]
        cv.imwrite(self.save_path + basename, label2RGB(predict))
        if self.label_path:
            label = cv.imread(self.label_path + basename, cv.IMREAD_GRAYSCALE)
            return cal_confu_matrix(label=label, predict=predict, class_num=self.class_num)

    def predict(self):
        model = self.model
        weighs = ['E:/dpy/TZBdeformable/weight/300_weights_2019-10-21-13-10-04.h5', ]
        save_path = self.save_path
        for weigh in weighs:
            kn = weigh.split('_')[-1][:6]
            self.save_path = save_path + 'predict' + kn + '/'
            os.makedirs(self.save_path, exist_ok=True)
            model.load(weigh)

            img_pathes = glob(self.img_path + '*.tif')
            count = 0
            confu_matrix = np.zeros((self.class_num, self.class_num), dtype=np.int32)
            for img_path in img_pathes:
                print('正在预测: ' + os.path.basename(img_path) + '  (total: ' + str(len(img_pathes)) +
                      '  finished:' + str(count) + ')')

                if self.label_path:
                    confu_matrix += self.predict_image_overlap_rate(model, img_path=img_path)
                else:
                    self.predict_image_overlap_rate(model, img_path=img_path)
                count += 1

            print('完成预测！')
            if self.label_path:
                metrics(confu_mat_total=confu_matrix, save_path=self.save_path)


if __name__ == '__main__':
    start_time = time.time()
    print('进行预测...')

    img_path = "F:\\zl_datasets\\TZBsubject2dataset\\val&test\\img\\"
    save_path= "F:\\zl_datasets\\TZBsubject2dataset\\val&test\\resultdpy\\"
    label_path = "F:\\zl_datasets\\TZBsubject2dataset\\val&test\\label\\"

    model = myUnet()
    # model.load(r"E:\dpy\TZBdeformable\weight\300_weights_2019-10-21-13-10-04.h5")

    '''method2'''

    patch_size = 640
    overlap_rate = 1/4
    test = TestFullImage(img_path=img_path, model=model, save_path=save_path, class_num=5,
                         patch_size=patch_size, overlap_rate=overlap_rate, label_path=label_path)  #

    test.predict()
    during_time = time.time() - start_time
    print('测试时间：%d(minute)%.2f(second)' % (during_time//60, (during_time-during_time//60*60)))

'''
640-1/2
0(minute)51.33(second) 68.3109
640-1/4
0(minute)35.36(second) 68.2618
640-1/8
0(minute)32.82(second) 67.9164
640-1/16
0(minute)32.53(second) 67.9951
640-1/32
0(minute)29.09(second) 67.9382  

512-1/8
0(minute)37.60(second) 67.7967

640-1/4
0(minute)35.36(second) 68.2618
512-1/4
0(minute)43.78(second) 67.9687
256-1/4
1(minute)34.84(second) 67.0281
patch_size越大，用时越少，但占用显存更多；
重叠率越大，精度提升越高；当超过某个值时，提升的效果不明显。
'''