import numpy as np
import caffe
import cv2
from PIL import Image

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
  image = image.copy()              # don't modify destructively
  image = image.squeeze()           # get rid of extra dimensions
  image = image[::-1]               # BGR -> RGB
  image = image.transpose(1, 2, 0)  # CHW -> HWC
  image[:, :, 0] += 103.939
  image[:, :, 1] += 116.779
  image[:, :, 2] += 123.68

  # clamp values in [0, 255]
  image[image < 0], image[image > 255] = 0, 255

  # round and cast from float32 to uint8
  image = np.round(image)
  image = np.require(image, dtype=np.uint8)

  return image

caffe.set_device(0)
caffe.set_mode_gpu()

# load the models
descriptor = caffe.Net('./models/descriptor/descriptor.prototxt', caffe.TRAIN)
texture_net_solver = caffe.get_solver('./solver.prototxt') # generator + descriptor

# fine-tune on VGG portion (descriptor) of texture_net
descriptor.copy_from('./models/descriptor/VGG_ILSVRC_19_layers.caffemodel')
texture_net_solver.net.copy_from('./models/descriptor/VGG_ILSVRC_19_layers.caffemodel')

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': descriptor.blobs['input1'].data.shape})
transformer.set_mean('data', np.array([103.939,116.779,123.68])) # imagenet mean
transformer.set_transpose('data', (2,0,1))    # HWC -> CHW
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR
transformer.set_raw_scale('data', 255.0)      # [0,1] -> [0,255]

# load the image in the data layer
im = caffe.io.load_image('./data/textures/red-peppers256.o.jpg')

# subtract imagenet mean
#im[:, :, 0] -= 103.939
#im[:, :, 1] -= 116.779
#im[:, :, 2] -= 123.68

# resize to target size
im = cv2.resize(im, (256,256), interpolation=cv2.INTER_CUBIC)

# preprocess target texture and set descriptor input
descriptor.blobs['input1'].data[...] = transformer.preprocess('data', im)

# compute initial pass
descriptor.forward()

# set targets
texture_net_solver.net.blobs['texture_target1_1'].data[...] = descriptor.blobs['gram1_1'].data[...].copy()
texture_net_solver.net.blobs['texture_target2_1'].data[...] = descriptor.blobs['gram2_1'].data[...].copy()
texture_net_solver.net.blobs['texture_target3_1'].data[...] = descriptor.blobs['gram3_1'].data[...].copy()
texture_net_solver.net.blobs['texture_target4_1'].data[...] = descriptor.blobs['gram4_1'].data[...].copy()
#texture_net_solver.net.blobs['content_target4_2'].data[...] = descriptor.blobs['conv4_2'].data[...].copy()
texture_net_solver.net.blobs['texture_target5_1'].data[...] = descriptor.blobs['gram5_1'].data[...].copy()

# learn texture
#texture_net_solver.solve()
