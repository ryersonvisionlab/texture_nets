#include <vector>

#include "caffe/layers/noise_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void NoiseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  
  // Fetch params
  batch_size_ = this->layer_param_.noise_data_param().batch_size();
  channels_ = this->layer_param_.noise_data_param().channels();
  spatial_size_ = this->layer_param_.noise_data_param().spatial_size();
  distribution_ = this->layer_param_.noise_data_param().distribution();
  mu_ = this->layer_param_.noise_data_param().mu();
  sigma_ = this->layer_param_.noise_data_param().sigma();
  size_ = channels_ * spatial_size_ * spatial_size_;
  
  // Preliminary checks
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, and spatial_size must be specified and"
      " positive in noise_data_param";
  CHECK_GT(sigma_, 0) <<
      "sigma must be specified and positive in noise_data_param";
  CHECK(distribution_ == "uniform" || distribution_ == "gaussian") <<
      "distribution must be specified and be either \"uniform\" or \"gaussian\"";
  
  // Initialize shapes
  top[0]->Reshape(batch_size_, channels_, spatial_size_, spatial_size_);
  added_data_.Reshape(batch_size_, channels_, spatial_size_, spatial_size_);
  
  data_ = NULL;
  
  // Sync with added_data on gpu
  added_data_.cpu_data();
}

template <typename Dtype>
void NoiseDataLayer<Dtype>::AddBlob(const vector<Blob<Dtype>*> datum_vector) {
  size_t num = datum_vector.size();
  
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  CHECK_GT(num, 0) <<
      "There is no datum to add.";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  
  // Some necessary reshaping before anything else
  added_data_.Reshape(num, channels_, spatial_size_, spatial_size_);

  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, num);
  has_new_data_ = true;
}

template <typename Dtype>
void NoiseDataLayer<Dtype>::Reset(Dtype* data, int n) {
  CHECK(data);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  //if (this->layer_param_.has_transform_param()) {
  //  LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  //}
  data_ = data;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void NoiseDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "NoiseDataLayer needs to be initialized by calling Reset";
  
  top[0]->Reshape(batch_size_, channels_, spatial_size_, spatial_size_);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  
  pos_ = (pos_ + batch_size_) % n_;
  if (pos_ == 0)
    has_new_data_ = false;
}

INSTANTIATE_CLASS(NoiseDataLayer);
REGISTER_LAYER_CLASS(NoiseData);

}  // namespace caffe
