import os

batch_size = 8
channel = 3
img_size = (128, 128)

img_shape = (img_size[0], img_size[1], channel)
output_size = img_shape[0] * img_shape[1] * img_shape[2]

n_bottleneck = round(float(img_shape[0]) / 2.0)

f_whole_samples = os.path.join('data', 'malaria_dataset')
f_small_samples = os.path.join('data', 'small_samples')
