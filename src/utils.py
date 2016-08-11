import caffe
from caffe import layers as L, params as P

def block(bottom, num_output, k, id):
  # A convolution block that consists of:
  #  1. Circular padding
  #  2. kxk convolution with num_output number of filters
  #  3. Batch Normalization
  #  4. Leaky RelU activation
  block = {}
  block['block_conv' + id] = L.Convolution(bottom, convolution_param={'kernel_size':k,
                                                                'stride':1,
                                                                'num_output':num_output,
                                                                'pad':(k-1)/2})
  block['block_bn' + id] = L.BatchNorm(block['block_conv' + id], param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])
  block['block_relu' + id] = L.ReLU(block['block_bn' + id], in_place=True)

  return block

def join(bottom_small, bottom_large, scale, id):
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
  
  append(bottom_small, {'join_bn' + id: L.BatchNorm(bottom_small, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])})
  append(bottom_large, {'join_bn' + id: L.BatchNorm(bottom_large, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])})
  
  join['join_concat' + id] = L.Concat(bottom=[bottom_small, bottom_large], concat_param={'axis': 1})

  return join


# Helper function for appending layers to a Caffe net spec
def append(net_spec, layers):
  for top, layer in layers.iteritems():
    if hasattr(net_spec, top):
      print('Layer with same top already exists!')
      continue
    setattr(net_spec, top, layer)
