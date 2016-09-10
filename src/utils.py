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

def conv_relu(bottom, num_output, k, id):
  # A convolution followed by a RelU
  conv = L.Convolution(bottom, convolution_param={'kernel_size': k,
                                                  'num_output': num_output,
                                                  'pad': 1})
  relu = L.ReLU(conv, in_place=True)
  
  conv_relu = OrderedDict([('conv' + id, conv), ('conv_relu' + id, relu)])

  top = conv_relu['conv_relu' + id]

  return conv_relu, top

def max_pool(bottom, stride, k, id):
  pool = {'pool' + id: L.Pooling(bottom, kernel_size=k, stride=stride, pool=P.Pooling.MAX)}

  top = pool['pool' + id]

  return pool, top

# named fc_ to avoid function name conflict with whatever was causing it
def fc_(bottom, num_output, id):
  fc = {'fc' + id: L.InnerProduct(bottom, inner_product_param={'num_output': num_output})}
  
  top = fc['fc' + id]

  return fc, top

def fc_relu_drop(bottom, num_output, ratio, id):
  fc = L.InnerProduct(bottom, inner_product_param={'num_output': num_output})
  relu = L.ReLU(fc, in_place=True)
  drop = L.Dropout(relu, dropout_param={'dropout_ratio': ratio})

  fc_relu_drop = OrderedDict([('fc' + id, fc), ('fc_relu' + id, relu), ('fc_drop' + id, drop)])

  top = fc_relu_drop['fc_drop' + id]

  return fc_relu_drop, top

def softmax(bottom, id):
  softmax = {'prob' + id: L.Softmax(bottom)}
  
  top = softmax['prob' + id]
  
  return softmax, top

def texture_loss(bottom, target_dim, norm, id):
  gramian = L.Gramian(bottom, gramian_param={'normalize_output': norm})
  _, target = input(target_dim, id)
  euclidean_loss = L.EuclideanLoss(bottom=['gram' + id, 'target' + id])
  
  texture_loss = OrderedDict([('gram' + id, gramian), 
                              ('target' + id, target),
                              ('euclidean_loss' + id, euclidean_loss)])

  top = texture_loss['euclidean_loss' + id]

  return texture_loss, top

def input(dim, id):
  input = {'input' + id: L.Input(input_param={'shape': {
                                                'dim': dim}})}
  top = input['input' + id]

  return input, top
