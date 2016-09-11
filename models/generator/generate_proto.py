import sys
import caffe
from caffe import layers as L, params as P
sys.path.insert(0, '../../src')
from utils import *

def generate():
  # generate the generator prototxt
  ratios = [16, 8, 4, 2, 1]
  conv_num = 8
  
  conv_block_counter = 1;
  data_counter = 1;
  join_block_counter = 1;

  for i in range(0, len(ratios)):

    ns = caffe.NetSpec()
    
    data_key = 'data' + str(data_counter)   
    data = {data_key: L.NoiseData(batch_size=1,
                                  spatial_size=256/ratios[i],
                                  channels=3,
                                  min=-1,
                                  max=1,
                                  distribution='uniform')}
    data_counter = append(ns, data, data_counter)
 
    conv_block, top = block(getattr(ns, data_key), conv_num, 3, str(conv_block_counter))
    conv_block_counter = append(ns, conv_block, conv_block_counter)

    conv_block, top = block(top, conv_num, 3, str(conv_block_counter))
    conv_block_counter = append(ns, conv_block, conv_block_counter)

    conv_block, top = block(top, conv_num, 1, str(conv_block_counter))
    conv_block_counter = append(ns, conv_block, conv_block_counter)
    
    if i == 0:
      cur = ns
      cur_top = top
    else:
      join_block, cur_top = join(cur, cur_top, ns, top, conv_num*i, str(join_block_counter))
      append(cur, ns.tops)
      join_block_counter = append(cur, join_block, join_block_counter)
      
      conv_block, cur_top = block(cur_top, conv_num*(i+1), 3, str(conv_block_counter))
      conv_block_counter = append(cur, conv_block, conv_block_counter)

      conv_block, cur_top = block(cur_top, conv_num*(i+1), 3, str(conv_block_counter))
      conv_block_counter = append(cur, conv_block, conv_block_counter)

      conv_block, cur_top = block(cur_top, conv_num*(i+1), 1, str(conv_block_counter))
      conv_block_counter = append(cur, conv_block, conv_block_counter)

      if i == len(ratios)-1:
        conv_block, cur_top = block(cur_top, 3, 1, str(conv_block_counter))
        conv_block_counter = append(cur, conv_block, conv_block_counter)
   
  ns = cur

  with open('generator.prototxt', 'w') as W:
    W.write('%s\n' % ns.to_proto())


def main(argv):
  generate()


if __name__ == '__main__':
  main(sys.argv[1:])
