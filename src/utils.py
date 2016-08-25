import caffe
from caffe import layers as L, params as P
from collections import OrderedDict

# Helper function for appending layers to a Caffe net spec
def append(net_spec, layers, counter=0):
  for top, layer in layers.iteritems():
    if hasattr(net_spec, top):
      print(top + ': ' + 'Layer with same top already exists!')
      continue
    setattr(net_spec, top, layer)
  return counter + 1

def block(bottom, num_output, k, id):
  # A convolution block that consists of:
  #  1. Circular padding
  #  2. kxk convolution with num_output number of filters
  #  3. Batch Normalization
  #  4. RelU activation
  block_conv = L.Convolution(bottom, convolution_param={'kernel_size': k,
                                                        'stride': 1,
                                                        'num_output': num_output,
                                                        'pad': (k-1)/2,
                                                        'weight_filler': {
                                                          'type': 'xavier'
                                                        }})
  block_bn = L.BatchNorm(block_conv, use_global_stats=False, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])
  block_relu = L.ReLU(block_bn, in_place=True)
  
  block = OrderedDict([('block_conv' + id, block_conv), ('block_bn' + id, block_bn), ('block_relu' + id, block_relu)])
   
  top = block['block_relu' + id]

  return block, top

def join(ns_small, bottom_small, ns_large, bottom_large, num_output, id):
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

  join_small_nnup = L.Deconvolution(bottom_small, param={'lr_mult': 0, 'decay_mult': 0}, convolution_param={
                                                                                              'num_output': num_output,
                                                                                              'bias_term': False,
                                                                                              'pad': 0,
                                                                                              'kernel_size': 2,
                                                                                              'group': num_output,
                                                                                              'stride': 2,
                                                                                              'weight_filler': {
                                                                                                'type': 'constant',
                                                                                                'value': 1
                                                                                              }})
  join_small_bn = L.BatchNorm(join_small_nnup, use_global_stats=False, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])
  join_large_bn = L.BatchNorm(bottom_large, use_global_stats=False, param=[{'lr_mult': 0},{'lr_mult': 0},{'lr_mult': 0}])

  append(ns_small, OrderedDict([('join_small_nnup' + id, join_small_nnup), ('join_small_bn' + id, join_small_bn)]))
  append(ns_large, {'join_large_bn' + id: join_large_bn}) 

  join['join_concat' + id] = L.Concat(bottom=['join_small_bn' + id, 'join_large_bn' + id], concat_param={'axis': 1})
  
  top = join['join_concat' + id]

  return join, top
