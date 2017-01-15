# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:35:40 2016

@author: shiwei
"""
import skimage
import argparse
import sys
import h5py
import os
import pandas as pd
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description='use caffe to extract photo features and aggregate into business features')
    parser.add_argument('--caffe_root', type=str, default='/home/shiwei/caffe/',
                        help='the path to where caffe is installed')
    parser.add_argument('--caffe_mode', type=str, default='cpu',
                        help='choice of gpu or cpu, choose either gpu or cpu to use in caffe')
    parser.add_argument('--caffe_device', type=int, default=0,
                        help='choose which device to use in case you have multiple gpus')
    parser.add_argument('--model_def', type=str, default='models/msra_residual_net/ResNet-152-deploy.prototxt',
                        help='caffe model prototxt path relative to caffe root folder, default is ResNet 152')
    parser.add_argument('--model_weights', type=str, default='models/msra_residual_net/ResNet-152-model.caffemodel',
                        help='caffe model weights path relative to caffe root folder, default is ResNet 152')
    parser.add_argument('--mean_file', type=str, default='models/msra_residual_net/ResNet_mean.binaryproto',
                        help='the mean file associated with the pre-trained model')
    parser.add_argument('--layer', type=str, default='pool5',
                        help='the output layer you want to extract features from')
    parser.add_argument('--data_root', type=str, default='/media/shiwei/data/yelp/',
                        help='the folder where you store train and test images, \
                        train images should be stored under data_root/train_photos and \
                        test images should be stored under data_root/test_photos')
    parser.add_argument('--batch_size', type=int, default=14,
                        help='batch size for the net to extract features, deafult 14 is used under the setting of \
                        3*224*224 image and gtx 1080 8GB gpu')
    return parser.parse_args()

def preprocess(img, mu):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    out = resized_img * 255.0
    out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
    out -= mu
    return out.transpose((2,0,1))
    
def extract_features(images, mu, layer='pool5'):
    
    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,224,224)
    net.blobs['data'].data[...] = map(lambda x: preprocess(skimage.io.imread(x), mu), images)
    out = net.forward()

#     return net.blobs[layer].data
#     return np.hstack((net.blobs[x].data.flatten() for x in layers))     
    return net.blobs[layer].data
   
def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]
            
if __name__ == '__main__':
    args = parse_args()
    
    #setting up caffe
    caffe_root = args.caffe_root
    caffe_mode = args.caffe_mode
    caffe_device = args.caffe_device
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
    if caffe_mode == 'gpu':
        caffe.set_mode_gpu()
    elif caffe_mode == 'cpu':
        caffe.set_mode_gpu()
    else:
        print 'Error! Caffe must use either gpu or cpu to run!'
    caffe.set_device(caffe_device)  # if we have multiple GPUs, pick the fi  rst one
    model_def = caffe_root + args.model_def
    model_weights = caffe_root + args.model_weights
    #define the net
    net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
    #process mean file for pre-trained model
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( caffe_root + args.mean_file , 'rb' ).read()
    blob.ParseFromString(data)
    mean_bgr = caffe.io.blobproto_to_array(blob)[0]
    assert mean_bgr.shape == (3, 224, 224)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    
    batch_size = args.batch_size
    mu = mean_bgr.transpose((1,2,0))
    layer = args.layer
    data_root = args.data_root
    feature_width = net.blobs[layer].data.shape[1]
    
    '''
    
    PART I: extract train photo features
    
    '''
    
    #writing train image features, store in h5 format
    f = h5py.File(data_root+'train_image_res5c_features.h5','w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,feature_width), maxshape = (None,feature_width))
    f.close()
    
    #process train image file names
    train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
    train_folder = data_root+'train_photos/'
    train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['photo_id']]  # get full filename
    num_train = len(train_images)
    print "Number of training images: ", num_train
    
    #start to extract train image features
    for i in range(0,num_train, batch_size): 
        images = train_images[i: min(i+batch_size, num_train)]
        features = extract_features(images, mu)
        features = features.reshape((len(images),features.shape[1]))
        num_done = i+features.shape[0]
        f= h5py.File(data_root+'train_image_res5c_features.h5','r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        f['feature'].resize((num_done,features.shape[1]))
        f['feature'][i: num_done, :] = features
        f.close()
        if num_done%2000==0 or num_done==num_train:
            print "Train images processed: ", num_done
    
    '''
    
    PART II: extract test photo features
    
    '''
    #writing test image features, store in h5 format
    f = h5py.File(data_root+'test_image_res5c_features.h5','w')
    filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
    feature = f.create_dataset('feature',(0,feature_width), maxshape = (None,feature_width))
    f.close()
    
    
    test_photos = pd.read_csv(data_root+'test_photo_to_biz.csv')
    test_folder = data_root+'test_photos/'
    test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]  
    num_test = len(test_images)
    print "Number of test images: ", num_test
    
    # Test Images
    for i in range(0, num_test, batch_size): 
        images = test_images[i: min(i+batch_size, num_test)]
        features = extract_features(images, mu)
        features = features.reshape((len(images),features.shape[1]))
        num_done = i+features.shape[0]
        
        f= h5py.File(data_root+'test_image_res5c_features.h5','r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        f['feature'].resize((num_done,features.shape[1]))
        f['feature'][i: num_done, :] = features
        f.close()
        if num_done%2000==0 or num_done==num_test:
            print "Test images processed: ", num_done
    
    '''
    
    PART III: aggregate business features in train set
    
    '''
    train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
    train_labels = pd.read_csv(data_root+'train.csv').dropna()
    train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
    train_labels.set_index('business_id', inplace=True)
    biz_ids = train_labels.index.unique()
    print "Number of business: ", len(biz_ids) ,   "(4 business with missing labels are dropped)"
    
    ## Load image features
    f = h5py.File(data_root+'train_image_res5c_features.h5','r')
    image_features = np.copy(f['feature'])
    f.close()
    
    
    t= time.time()
    ## For each business, compute a feature vector 
    df = pd.DataFrame(columns=['business','label','feature vector'])
    index = 0
    for biz in biz_ids:  
        
        label = train_labels.loc[biz]['labels']
        image_index = train_photo_to_biz[train_photo_to_biz['business_id']==biz].index.tolist()
        folder = data_root+'train_photo_folders/'  
        
        features = image_features[image_index]
        mean_feature =list(np.mean(features,axis=0))
    
        df.loc[index] = [biz, label, mean_feature]
        index+=1
        if index%1000==0:
            print "Buisness processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
    
    with open(data_root+"train_biz_res5c_features.csv",'w') as f:  
        df.to_csv(f, index=False)
    
    
    '''
    
    PART IV: aggregate business features in test set
    
    '''
    test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
    biz_ids = test_photo_to_biz['business_id'].unique()
    
    ## Load image features
    f = h5py.File(data_root+'test_image_res5c_features.h5','r')
    image_filenames = list(np.copy(f['photo_id']))
    image_filenames = [name.split('/')[-1][:-4] for name in image_filenames]  #remove the full path and the str ".jpg"
    image_features = np.copy(f['feature'])
    f.close()
    print "Number of business: ", len(biz_ids)
    
    df = pd.DataFrame(columns=['business','feature vector'])
    index = 0
    t = time.time()
    
    for biz in biz_ids:     
        
        image_ids = test_photo_to_biz[test_photo_to_biz['business_id']==biz]['photo_id'].tolist()  
        image_index = [image_filenames.index(str(x)) for x in image_ids]
         
        folder = data_root+'test_photo_folders/'            
        features = image_features[image_index]
        # take the everage of all the photos per business
        mean_feature =list(np.mean(features,axis=0))
    
        df.loc[index] = [biz, mean_feature]
        index+=1
        if index%1000==0:
            print "Buisness processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
    
    with open(data_root+"test_biz_res5c_features.csv",'w') as f:  
        df.to_csv(f, index=False)