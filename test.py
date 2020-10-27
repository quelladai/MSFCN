import gc
from copy import deepcopy
import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import matplotlib
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from model import STSCNN
from modelUnet import OriginalUNet
from model_Unet_spatial import Spatial_Unet
from Indicators import mPSNR, mSSIM, SAM, CC



MAX_BATCH_SIZE = 128


masked = cv2.imread('E:/project/my/test/simulated/masked5.tif')
masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
masked = cv2.resize(masked, (512, 512))

temporal = cv2.imread('E:/project/my/test/simulated/temporal.tif')
temporal = cv2.cvtColor(temporal, cv2.COLOR_BGR2RGB)
temporal = cv2.resize(temporal, (512, 512))

# original = cv2.imread('E:/project/my/test/simulated/original.tif')
# original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
# original = cv2.resize(original, (512, 512))

class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))
            gc.collect()
            yield ori

datagen = DataGenerator(
    rescale=1. / 255,
)

batch_masked = np.stack([masked for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_masked1 = np.stack([masked1 for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_masked2 = np.stack([masked2 for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_masked3 = np.stack([masked3 for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_masked4 = np.stack([masked4 for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_masked5 = np.stack([masked5 for _ in range(MAX_BATCH_SIZE)], axis=0)
batch_temporal = np.stack([temporal for _ in range(MAX_BATCH_SIZE)], axis=0)
# batch_original = np.stack([original for _ in range(MAX_BATCH_SIZE)], axis=0)

generator_masked = datagen.flow(x=batch_masked, batch_size=1)
# generator_masked1 = datagen.flow(x=batch_masked1, batch_size=1)
# generator_masked2 = datagen.flow(x=batch_masked2, batch_size=1)
# generator_masked3 = datagen.flow(x=batch_masked3, batch_size=1)
# generator_masked4 = datagen.flow(x=batch_masked4, batch_size=1)
# generator_masked5 = datagen.flow(x=batch_masked5, batch_size=1)

generator_temporal = datagen.flow(x=batch_temporal, batch_size=1)
# generator_original = datagen.flow(x=batch_original, batch_size=1)

masked = next(generator_masked)
# masked1 = next(generator_masked1)
# masked2 = next(generator_masked2)
# masked3 = next(generator_masked3)
# masked4 = next(generator_masked4)
# masked5 = next(generator_masked5)
temporal = next(generator_temporal)
# original = next(generator_original)

model = Spatial_Unet()
model.load(r"E:\project\my\log\weight_spatial_Unet\New\1500_weights_2018-09-06-17-17-05.h5")
pred_img = model.predict([masked, temporal])
# pred_img1 = model.predict([masked1, temporal])
# pred_img2 = model.predict([masked2, temporal])
# pred_img3 = model.predict([masked3, temporal])
# pred_img4 = model.predict([masked4, temporal])
# pred_img5 = model.predict([masked5, temporal])

# model2 = OriginalUNet()
# model2.load(r"E:\project\my\log\weight\118032\Unet\1500_weights_2018-08-29-21-01-00.h5")
# pred_img2 = model2.predict(temporal)

# psnr = mPSNR(original[0,:,:,:], pred_img[0,:,:,:])
# ssim = mSSIM(original[0,:,:,:], pred_img[0,:,:,:])
# sam = SAM(original[0,:,:,:], pred_img[0,:,:,:])
# cc = CC(original[0,:,:,:], pred_img[0,:,:,:])
#
# psnr1 = mPSNR(original[0,:,:,:], pred_img1[0,:,:,:])
# ssim1 = mSSIM(original[0,:,:,:], pred_img1[0,:,:,:])
# sam1 = SAM(original[0,:,:,:], pred_img1[0,:,:,:])
# cc1 = CC(original[0,:,:,:], pred_img1[0,:,:,:])
#
# psnr2 = mPSNR(original[0,:,:,:], pred_img2[0,:,:,:])
# ssim2 = mSSIM(original[0,:,:,:], pred_img2[0,:,:,:])
# sam2 = SAM(original[0,:,:,:], pred_img2[0,:,:,:])
# cc2 = CC(original[0,:,:,:], pred_img2[0,:,:,:])
#
# psnr3 = mPSNR(original[0,:,:,:], pred_img3[0,:,:,:])
# ssim3 = mSSIM(original[0,:,:,:], pred_img3[0,:,:,:])
# sam3 = SAM(original[0,:,:,:], pred_img3[0,:,:,:])
# cc3 = CC(original[0,:,:,:], pred_img3[0,:,:,:])
#
# psnr4 = mPSNR(original[0,:,:,:], pred_img4[0,:,:,:])
# ssim4 = mSSIM(original[0,:,:,:], pred_img4[0,:,:,:])
# sam4 = SAM(original[0,:,:,:], pred_img4[0,:,:,:])
# cc4 = CC(original[0,:,:,:], pred_img4[0,:,:,:])
#
# psnr5 = mPSNR(original[0,:,:,:], pred_img5[0,:,:,:])
# ssim5 = mSSIM(original[0,:,:,:], pred_img5[0,:,:,:])
# sam5 = SAM(original[0,:,:,:], pred_img5[0,:,:,:])
# cc5 = CC(original[0,:,:,:], pred_img5[0,:,:,:])


#
# psnr2 = mPSNR(masked[0,:,:,:], pred_img2[0,:,:,:])
# ssim2 = mSSIM(masked[0,:,:,:], pred_img2[0,:,:,:])
# sam2 = SAM(masked[0,:,:,:], pred_img2[0,:,:,:])
# cc2 = CC(masked[0,:,:,:], pred_img2[0,:,:,:])
#
# print('mPSNR:',psnr1,psnr2,psnr3,psnr4,psnr5)
# print('mSSIM:',ssim1,ssim2,ssim3,ssim4,ssim5)
# print('SAM:',sam1,sam2,sam3,sam4,sam5)
# print('CC:',cc1,cc2,cc3,cc4,cc5)

# psnr = mPSNR(masked[0,:,:,:], temporal[0,:,:,:])
# ssim = mSSIM(masked[0,:,:,:], temporal[0,:,:,:])
# sam = SAM(masked[0,:,:,:], temporal[0,:,:,:])
# cc = CC(masked[0,:,:,:], temporal[0,:,:,:])
#
# print('mPSNR:',psnr)
# print('mSSIM:',ssim)
# print('SAM:',sam)
# print('CC:',cc)


# Show side by side
# _, axes = plt.subplots(1, 4, figsize=(30, 5))
#
# axes[0].imshow(masked[0,:,:,:])
# axes[1].imshow(temporal[0,:,:,:])
# axes[2].imshow(pred_img[0,:,:,:])
# axes[3].imshow(original[0,:,:,:])
# axes[0].set_title('Masked Image')
# axes[1].set_title('Temporal Image')
# axes[2].set_title('Predicted Image')
# axes[3].set_title('Original Image')
# plt.savefig(r'E:\project\my\test\simulated\epoch1500_simulated_masked1_img.png')
# plt.show()

# _, axes = plt.subplots(1, 3, figsize=(30, 5))
#
# axes[0].imshow(masked[0,:,:,:])
# axes[1].imshow(temporal[0,:,:,:])
# axes[2].imshow(pred_img[0,:,:,:])
# axes[0].set_title('Masked Image')
# axes[1].set_title('Temporal Image')
# axes[2].set_title('Predicted Image')
# plt.savefig(r'E:\project\my\test\simulated\epoch1500_real_masked137_img.png')
# plt.show()

img = pred_img[0,:,:,:]
matplotlib.image.imsave('E:/project/my/test/simulated/predicted5.png', img)



