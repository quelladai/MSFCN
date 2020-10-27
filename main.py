import matplotlib
matplotlib.use('Agg')

from modelUnetD import myUnet
import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import generatedata_0riginal,generatedata_0riginal_1,generatedata
import numpy as np
import glob
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


BATCH_SIZE = 2

# Train_DIR = "E:\\Experiment data\\data\\Landsat\\Train\\"
# Val_DIR = "E:\\Experiment data\\data\\Landsat\\Val\\"
# Test_DIR = "E:\\Experiment data\\data\\Landsat\\Test\\"
# print('Start  training')
#
Train_DIR = "E:\\Experiment data\\data\\Landsat\\train\\"
Val_DIR = "E:\\Experiment data\\data\\Landsat\\val\\"
Test_DIR = "E:\\Experiment data\\data\\Landsat\\test\\"

# print('path ')
# if not os.path.exists(Test_DIR):
#     os.makedirs(Test_DIR)

train_generator = generatedata_0riginal(Train_DIR, BATCH_SIZE)
val_generator = generatedata_0riginal(Val_DIR, BATCH_SIZE)
test_generator = generatedata_0riginal(Test_DIR, BATCH_SIZE)

test = next(test_generator)
(masked, label) = test
print(masked.shape)
print(label.shape)

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
            if (img[i][j][0]>=0.5) and (img[i][j][0]>=img[i][j][1]):
                x[i][j][0] = 1 #红色 阴影
                x[i][j][1] = 0
                x[i][j][2] = 0
            elif (img[i][j][1]>=0.5) and (img[i][j][0]<img[i][j][1]):
                x[i][j][0] = 0
                x[i][j][1] = 1 #云

                x[i][j][2] = 0
            else:
                x[i][j][0] = 0
                x[i][j][1] = 0
                x[i][j][2] = 1
    return x

patch_size = 512

def plot_callback(model):

    # Get samples & Display them
    mask = model.predict(masked)
    (b, h, w, c) = mask.shape
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    label_normal = np.ndarray((b, h, w, c), dtype=np.uint8)

    # Clear current output and display test images
    # mask_show = np.zeros([b, h, w, 3*c])
    for i in range(len(masked)):
        labelimg = mask[i]
        label_normal[i] = max_img(labelimg)
        # print(label_normal.shape)
        # mask[i, :, :, :][mask[i, :, :, :]> 0.8] = 1
        # mask[i, :, :, :][mask[i, :, :, :] <= 0.8] = 0
        # mask_array = mask[i, :, :, :]
        # mask_array[mask_array >= 0.5] = 1
        # mask_array[mask_array < 0.5] = 0
        # mask_img = array_to_img(mask[i, :, :, :])
        # mask_img.save(r'E:\project\MSCN\log\testsample\result_{}_{}.jpg'.format(i, pred_time))
        # mask_show = np.zeros([b, h, w, c])
        # print(label_normal.shape)
        # mask_show[i, :, :, :] = np.dstack((label_normal[i, :, :], label_normal[i, :, :], label_normal[i, :, :]))
        # mask_img = array_to_img(mask_show[i, :, :, :] * 255)
        # mask_img.save(r'E:\project\UNet-detection\log\classification\testsample\result_{}_{}.jpg'.format(i, pred_time))
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        print(masked.shape)
        # axes[0].imshow(masked[i, :, :, 1:4])
        axes[0].imshow(masked[i, :, :, :])
        # mask_show[i, :, :, :] = np.dstack((mask[i, :, :, :], mask[i, :, :, :], mask[i, :, :, :]))
        axes[1].imshow(label_normal[i, :, :, :] * 255)
        axes[2].imshow(label[i, :, :, :] * 255)
        # axes[1].imshow(mask_show[i, :, :, :] * 255)
        # axes[2].imshow(label_show[i, :, :, :] * 255)
        axes[0].set_title('Masked Image')
        axes[1].set_title('mask Image')
        axes[2].set_title('mask_label Image')

        plt.savefig(r'H:\Our_cloudshao\test_samples\img_{}_{}.png'.format(i, pred_time))
        plt.close()

model = myUnet(weight_filepath='H:/Our_cloudshao/weight/')
model.summary()
#
# model.load(r"H:\GF_UNet\weight\76_weights_2019-12-27-17-58-31.h5")

img_num = len(glob.glob(Train_DIR + "masked\\*.tif"))
val_num = len(glob.glob(Val_DIR + "masked\\*.tif"))


checkpoint_fn = os.path.join(
    'H:\\Our_cloudshao\\bestweight\\checkpoint-{epoch:02d}-val_acc_{val_acc:.2f}.h5')
checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_acc', mode='auto', save_best_only='True')
tensorboard = TensorBoard(log_dir='H://Our_cloudshao/logs', write_graph=True)
callbacks_list = [checkpoint, tensorboard]
# callbacks_list = [checkpoint, tensorboard]

model.fit(
    train_generator,
    steps_per_epoch=img_num // BATCH_SIZE,
    # steps_per_epoch=2,
    validation_data=val_generator,
    validation_steps=val_num //BATCH_SIZE,
    # validation_steps=2,
    epochs=1000,
    plot_callback=plot_callback,
    callbacks=callbacks_list
)


# model.fit(
#     train_generator,
#     steps_per_epoch=2,
#     validation_data=val_generator,
#     validation_steps=2,
#     epochs=5000,
#     plot_callback=plot_callback,
#     callbacks=[
#         TensorBoard(log_dir='E:/project/UNet-detection/log/logs/Unetfintune', write_graph=False)
#     ]
# )