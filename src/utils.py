import caffe
from caffe import layers as L, params as P
from collections import OrderedDict

# Helper function for appending layers to a Caffe net spec
def append(net_spec, layers):
  for top, layer in layers.iteritems():
    if hasattr(net_spec, top):
      print('Layer with same top already exists!')
      continue
    setattr(net_spec, top, layer)

def block(bottom, num_output, k, id):
  # A convolution block that consists of:
  #  1. Circular padding
  #  2. kxk convolution with num_output number of filters
  #  3. Batch Normalization
  #  4. Leaky RelU activation
  block_conv = L.Convolution(bottom, convolution_param={'kernel_size':k,
                                                        'stride':1,
                                                        'num_output':num_output,
                                                        'pad':(k-1)/2})
  block_bn = L.BatchNorm(block_conv, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])
  block_relu = L.ReLU(block_bn, in_place=True)
  
  block = OrderedDict([('block_conv' + id, block_conv), ('block_bn' + id, block_bn), ('block_relu' + id, block_relu)])
   
  top = block['block_relu' + id]

  return block, top

def join(ns_small, bottom_small, ns_large, bottom_large, scale, id):
  # Joining two tensors (a smaller scale tensor with a larger scale tensor) by:
  #  Smaller tensor:
  #    1. Spatial upsampling using nearest neighbour with a stride of 2
  #    2. Batch Normalization
  #  Larger tensor:
  #    1. Batch Normalization
  #  
  #  Final tensor:
  #    1. Depth concatenation of upsampled smaller scale tensor onto larger scale tensor
  join = {}

  append(ns_small, {'join_small_bn' + id: L.BatchNorm(bottom_small, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])})
  append(ns_large, {'join_large_bn' + id: L.BatchNorm(bottom_large, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])}) 

  join['join_concat' + id] = L.Concat(bottom=['join_small_bn' + id, 'join_large_bn' + id], concat_param={'axis': 1})
  
  top = join['join_concat' + id]

  return join, top
