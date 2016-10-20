import numpy as np
import caffe
from PIL import Image

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
  image = image.copy()              # don't modify destructively
  image = image.squeeze()           # get rid of extra dimensions
  image = image[::-1]               # BGR -> RGB
  image = image.transpose(1, 2, 0)  # CHW -> HWC
  image[:, :, 2] += 103.939
  image[:, :, 1] += 116.779
  image[:, :, 0] += 123.68

  # clamp values in [0, 255]
  image[image < 0], image[image > 255] = 0, 255

  # round and cast from float32 to uint8
  image = np.round(image)
  image = np.require(image, dtype=np.uint8)

  return image

caffe.set_device(0)
caffe.set_mode_gpu()

# load the model
generator = caffe.Net('./models/generator/generator.prototxt', caffe.TRAIN)
generator.copy_from('./_iter_1500.caffemodel')

# compute initial pass
generator.forward()

# save image
num = generator.blobs['conv1'].data.shape[0]
for i in range(0, num):
  img = deprocess_net_image(generator.blobs['conv1'].data[i])
  Image.fromarray(img).save(str(i+1) + '.png')
