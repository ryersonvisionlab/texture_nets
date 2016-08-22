#!/home/mtesfald/anaconda2/bin/python

import numpy as np

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

import matplotlib.pyplot as plt

net = caffe.Net('./deploy.prototxt', caffe.TEST)

out = net.forward()

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image.squeeze()           # get rid of extra dimensions
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

img = deprocess_net_image(out['upsample'])

plt.imshow(img)
plt.show()
