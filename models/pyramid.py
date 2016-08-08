def create(caffe, args):
  from caffe import layers as L
  from caffe import params as P
  
  ratios = [1] # keep it at a single scale for now
  
  num_conv = 8 # number of filters

  cur = None
  for i in range(1, len(ratios)+1):
    net = caffe.NetSpec()
    # TODO: create our own data layer that generates a mini-batch of noise tensors. Look at GenNoise and NoiseFill from 
    # texture net
    net.data, net.label = L.Data(batch_size=args['batch_size'], source=args['data'], ntop=2)
