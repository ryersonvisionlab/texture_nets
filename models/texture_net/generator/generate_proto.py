import sys
import caffe
from caffe import layers as L, params as P
sys.path.insert(0, '../../../src')
from utils import block, join, append

def deploy():
  # generate the generator prototxt for only feed-forward
  pass

def solver():
  # generate the generator prototxt for the solver
  pass

def train_val():
  # generate the generator prototxt for training and validation
  ratios = [32, 16]
  conv_num = 8
  
  for i in range(0, len(ratios)):

    ns = caffe.NetSpec()
    
    data_key = 'data' + str(i+1)   
    data = {data_key: L.Data(batch_size=1, source='./path/to/source')}
    append(ns, data)
    
    pool_key = 'pool' + str(i+1)
    pool = {pool_key: L.Pooling(getattr(ns, data_key), kernel_size=ratios[i], stride=ratios[i], pool=P.Pooling.AVE)}
    append(ns, pool)
 
    conv_block, top = block(getattr(ns, pool_key), conv_num, 3, str(3*i+1))
    append(ns, conv_block)

    conv_block, top = block(top, conv_num, 3, str(3*i+2))
    append(ns, conv_block)

    conv_block, top = block(top, conv_num, 1, str(3*i+3))
    append(ns, conv_block)
    
    if i == 0:
      cur = ns
      cur_top = top
    else:
      join_block, top = join(cur, cur_top, ns, top, i, str(i))
      append(cur, ns.tops)
      append(cur, join_block)
      
      conv_block, top = block(top, conv_num*(i+1), 3, str(3*i+4))
      append(cur, conv_block)

      conv_block, top = block(top, conv_num*(i+1), 3, str(3*i+5))
      append(cur, conv_block)

      conv_block, top = block(top, conv_num*(i+1), 1, str(3*i+6))
      append(cur, conv_block)

      if i == len(ratios)-1:
        conv_block, top = block(top, 3, 1, str(3*i+7))
        append(cur, conv_block)
   
  ns = cur

  with open('train_val.prototxt', 'w') as W:
    W.write('name: "TextureGeneratorNet"\n')
    W.write('%s\n' % ns.to_proto())


def main(argv):
  deploy()
  solver()
  train_val()


if __name__ == '__main__':
  main(sys.argv[1:])
