if [ ! -d "./vgg_ilsvrc_19_layers" ]; then
  mkdir vgg_ilsvrc_19_layers
fi

wget -P ./vgg_ilsvrc_19_layers -c https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt
wget -P ./vgg_ilsvrc_19_layers -c --no-check-certificate https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel
wget -P ./vgg_ilsvrc_19_layers -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
