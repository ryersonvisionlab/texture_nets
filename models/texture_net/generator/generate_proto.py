import sys
import caffe
from caffe import layers as L, params as P

def deploy():
  # generate the generator prototxt for only feed-forward
  pass

def solver():
  # generate the generator prototxt for the solver
  pass

def train_val():
  # generate the generator prototxt for training and validation
  ns = caffe.NetSpec()
  ns.conv = L.Convolution(bottom="yourmom", convolution_param={'kernel_size':1,
                                                     'stride':2,
                                                     'num_output':3, 
                                                     'pad':5},
                                  param=[{'lr_mult':1, 'decay_mult':1},
                                         {'lr_mult':2, 'decay_mult':0}],
                                  include={'phase': caffe.TRAIN})
  with open('train_val.prototxt', 'w') as W:
    W.write('name: "TextureGeneratorNet"\n')
    W.write('%s\n' % ns.to_proto())

def main(argv):
  deploy()
  solver()
  train_val()

if __name__ == '__main__':
  main(sys.argv[1:])
