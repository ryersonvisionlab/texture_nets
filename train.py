# train.py
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse, importlib
import caffe

def main(argv):
  # Parameters
  parser = argparse.ArgumentParser(description='Feed-forward synthesis of textures and stylized images.', version='0.1')
  
  parser.add_argument('--learningRate', default=1e-3, type=float)
  
  parser.add_argument('--numIterations', default=50000, type=int)
  parser.add_argument('--batchSize', default=1, type=int)
  
  parser.add_argument('--imageSize', default=256, type=int)

  parser.add_argument('--mode', default='texture')

  parser.add_argument('--checkpointsPath', default='data/checkpoints/', help='Directory to store intermediate results.')
  parser.add_argument('--model', default='pyramid', help='Path to generator model description file.')

  parser.add_argument('--normalizeGradients', default=False, help='L1 gradient normalization inside descriptor net.')

  parser.add_argument('--pretrainedProto', default='data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt')
  parser.add_argument('--pretrainedModel', default='data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
   

  try:
    args = vars(parser.parse_args())
  except IOError, msg:
    parser.error(str(msg))

 
  # Define models
  generator = importlib.import_module('models' + '.' + args['model']).create() 


if __name__ == '__main__':
  main(sys.argv[1:])
