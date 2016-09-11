import sys
import caffe
from caffe import layers as L, params as P
sys.path.insert(0, '../../src')
from utils import *

def generate(bottom=None):
  # generate the descriptor prototxt. It's VGG19 with gram layers
  # and euclidean loss layers inserted at certain places.

  input_dim = [16,3,256,256]

  data_counter = 1
  conv_relu_counter = 1
  pool_counter = 1
  fc_counter = 1
  softmax_counter = 1
  txt_loss_counter = 1

  ns = caffe.NetSpec()
  
  if not bottom:
    data, top = input(input_dim, str(data_counter))
    data_counter = append(ns, data, data_counter)

  for i in range(1, 10):
    if i <= 2:
      
      if i == 1:
        num_output = 64
      else:
        num_output = 128

      # convi_1, relui_1
      if bottom and i == 1:
        cr, top = conv_relu(bottom, num_output, 3, False, str(i) + '_1')
      else:
        cr, top = conv_relu(top, num_output, 3, False, str(i) + '_1')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      # attach texture loss (after relu1_1, relu2_1)
      loss, _ = texture_loss(top, [16,num_output**2], True, str(i) + '_1')
      txt_loss_counter = append(ns, loss, txt_loss_counter)

      # convi_2, relui_2
      cr, top = conv_relu(top, num_output, 3, False, str(i) + '_2')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      # pooli
      pl, top = max_pool(top, 2, 2, str(i))
      pool_counter = append(ns, pl, pool_counter)

    if i >= 3 and i <= 5:
      
      if i == 3:
        num_output = 256
      else:
        num_output = 512
      
      # convi_1, relui_1
      cr, top = conv_relu(top, num_output, 3, False, str(i) + '_1')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      if i <= 4:
        # attach texture loss (after relu3_1, relu4_1)
        loss, _ = texture_loss(top, [16,num_output**2], True, str(i) + '_1')
        txt_loss_counter = append(ns, loss, txt_loss_counter)

      # convi_2, relui_2
      cr, top = conv_relu(top, num_output, 3, False, str(i) + '_2')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      # convi_3, relui_3
      cr, top = conv_relu(top, num_output, 3, False, str(i) + '_3')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      # convi_4, relui_4
      cr, top = conv_relu(top, num_output, 3, False, str(i) + '_4')
      conv_relu_counter = append(ns, cr, conv_relu_counter)

      # pooli
      pl, top = max_pool(top, 2, 2, str(i))
      pool_counter = append(ns, pl, pool_counter)    

    if i >= 6 and i <= 7:
      # fci, relui, dropi
      fc, top = fc_relu_drop(top, 4096, 0.5, False, str(i))
      fc_counter = append(ns, fc, fc_counter)        

    if i == 8:
      # fc8
      fc, top = fc_(top, 1000, False, str(i))
      fc_counter = append(ns, fc, fc_counter)

    if i == 9:
      # prob
      prob, top = softmax(top, str(softmax_counter))
      softmax_counter = append(ns, prob, softmax_counter)

  if not bottom:
    with open('descriptor.prototxt', 'w') as W:
      W.write('%s\n' % ns.to_proto())
  else:
    with open('descriptor_merge.prototxt', 'w') as W:
      W.write('%s\n' % ns.to_proto())


def main(argv):
  generate()
  generate('block_bn28')


if __name__ == '__main__':
  main(sys.argv[1:])
