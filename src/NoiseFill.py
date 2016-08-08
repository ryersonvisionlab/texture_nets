import caffe
import numpy as np

class NoiseFillDataLayer(caffe.Layer):

  """
  This is a data layer which generates dim*dim*K noise-tensors
  from a uniform distribution
  """

  def setup(self, bottom, top):
    self.top_names = ['data', 'label']
    
    # === Read input parameters ===
    
    # params is a python dictionary with layer parameters
    params = eval(self.param_str)
    
    # check the params for validity
    check_params(params)
    
    self.batch_size = params['batch_size']

    # === Reshape tops ===
    # since we use a fixed input image size, we can shape the data layer
    # once. Else, we'd have to do it in the reshape call.
    

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    pass

  def backward(self, bottom, top):
    pass


def check_params(params):
  
