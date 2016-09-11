import numpy as np
import caffe
import cv2

caffe.set_device(0)
caffe.set_mode_gpu()

# load the models
descriptor = caffe.Net('./models/descriptor/descriptor.prototxt', caffe.TRAIN)
texture_net_solver = caffe.get_solver('./solver.prototxt') # generator + descriptor

# fine-tune on VGG portion (descriptor) of texture_net
texture_net_solver.net.copy_from('./models/descriptor/vgg_normalised.caffemodel')

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': descriptor.blobs['input1'].data.shape})
transformer.set_transpose('data', (2,0,1))    # HWC -> CHW
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR
transformer.set_raw_scale('data', 255.0)      # [0,1] -> [0,255]

# load the image in the data layer
im = caffe.io.load_image('./data/textures/cezanne.jpg')
im = cv2.resize(im, (256,256), interpolation=cv2.INTER_CUBIC)
descriptor.blobs['input1'].data[...] = transformer.preprocess('data', im)

# compute initial pass
descriptor.forward()

# set targets
texture_net_solver.net.blobs['target1_1'].data[...] = descriptor.blobs['gram1_1'].data[...]
texture_net_solver.net.blobs['target2_1'].data[...] = descriptor.blobs['gram2_1'].data[...]
texture_net_solver.net.blobs['target3_1'].data[...] = descriptor.blobs['gram3_1'].data[...]
texture_net_solver.net.blobs['target4_1'].data[...] = descriptor.blobs['gram4_1'].data[...]

# learn texture
texture_net_solver.solve()
