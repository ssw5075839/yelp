This is a solution to Kaggle Yelp Restaurant Photo Classification https://www.kaggle.com/c/yelp-restaurant-photo-classification

This solution achieves around 0.8241 in private leader board with ResNet-152. If you want to try to ensemble more CNNS, you could probably achieve higher scores.
In this competition, you are given photos that belong to a business and asked to predict the business attributes. There are 9 different attributes in this problem:

0: good_for_lunch

1: good_for_dinner

2: takes_reservations

3: outdoor_seating

4: restaurant_is_expensive

5: has_alcohol

6: has_table_service

7: ambience_is_classy

8: good_for_kids

These labels are annotated by the Yelp community. Your task is to predict these labels purely from the business photos uploaded by users. 

The evaluation metric for this competition is Mean F1-Score https://www.kaggle.com/wiki/MeanFScore

In this competition, each business could have various number of photos (vary from 2 to 2000) and each photo may reflect different labels. Some labels are extremely hard and vague to classify clearly even for humans, for example, takes reservations and ambience_is_classy. Therefore, it is not surprising that later on you would find some label's f1 score is much worse than others.

The basic idea of this solution is using CNN to extract photo features and then aggregate these features for each business. With these business features in hand, we can use our favorite classification algorithms like xgboost and svm to classify these nine labels.

In CNN feature extraction, we only forward CNN up to certain layer rather than all the way to the top because lower layer usually contains much more abstract information. For example, in ResNet-152, we forward CNN up to ‘pool5’ layer. This layer has dimension of 2048 and just below the last fc1000 layer. We believe this layer contains enough abstract information while that information is not too low-level to be useful.

Then for each business, all those 2048D features from each of its photos are aggregated to form the feature of this particular business. However, this aggregation process is not trivial. One simple method is just taking the average of all the 2048D features. Another fancier method, as in the 1st place solution http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/ indicated, is computing fisher vector for these 2048D features. However, for those fancy methods, one limiting factor is the number of photos for each business varies a lot, from 2 to 2000. For example, to compute fisher vector, how many components should we use in GMM? 2 may be too small since we have 9 labels here. 9 seem reasonable but a lot of businesses contain less than 9 photos and those precious training data have to be obsoleted.

Actually, the 1st place solution doesn’t show too much solid advantage over others. I would personally say its leading in score is due to its ensemble of many different CNNs. (0.8317 vs 0.8241)

After so many trail and errors, I discovered that a simple average is the most efficient way to aggregate the 2048D features and decide to use it unless others tell me other efficient way.

After business-level features are computed, we can use our favorite classifiers to deal with this problem. I use Xgboost and SVM to classify the 9 labels. One detail is that I find SVM may need PCA the 2048D features to 200D first. In this way I can achieve a little better score. Note that Xgboost doesn’t need PCA first because it has somewhat automatic feature selection cabability.

Another point is that we cannot train the CNN end to end because at the top the classifier (Xgboost) cannot back-propgated. Maybe later on we can incorporate classifier capable of back-propgate to the caffe framework and dealing with average of various number of input. If this goal is achieved, the fine tuned CNN is expected to achieve better performance

Detailed code running instruction:

Pre-request:

Python (Recommand Anaconda https://www.continuum.io/downloads)

caffe (http://caffe.berkeleyvision.org/)

Resnet (https://github.com/KaimingHe/deep-residual-networks)

Competetion data (https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)

1. Run CNN feature extractor: 

   python CNN_extract_features.py

In CNN_extract_features.py, the program will first extract train and test photo features using the selected CNN, then aggregate image level features into business level features.

optional arguments:
  -h, --help            show this help message and exit
  
  --caffe_root CAFFE_ROOT
                        the path to where caffe is installed. If you have already add CAFFE_ROOT in you enviroment variables, you probably don't need this argument. If you encounter problem when import caffe, double check if this path is correct.
                        
  --caffe_mode CAFFE_MODE
                        choice of gpu or cpu, choose either gpu or cpu to use in caffe. Default is 'cpu'.
                        
  --caffe_device CAFFE_DEVICE
                        choose which device to use in case you have multiple gpus. Default is '0'.
                        
  --model_def MODEL_DEF
                        caffe model prototxt path relative to caffe root folder, default is ResNet 152. It is recommended to put model prototxt under caffe_root/models.
                        
  --model_weights MODEL_WEIGHTS
                        caffe model weights path relative to caffe root folder, default is ResNet 152. It is recommended to put model weights under caffe_root/models.
                        
  --mean_file MEAN_FILE
                        the mean file associated with the pre-trained model, default is ResNet 152 (ResNet_mean.binaryproto).
                        
  --layer LAYER         the output layer you want to extract features from, default is 'pool5'. It is recommended to take a look at your CNN structure before choose this option. Ethereon netscope is a useful online CNN visualization tool for caffe. Simply paste your CNN prototxt into the editor and it will visualize the CNN structure for you. http://ethereon.github.io/netscope/#/editor
  
  --data_root DATA_ROOT
                        the folder where you store train and test images, train images should be stored under data_root/train_photos and test images should be stored under data_root/test_photos. Please unzip the dataset you download from kaggle into the folder as indicated.
                        
  --batch_size BATCH_SIZE
                        batch size for the CNN to extract features, deafult 14 is used under the setting of 3*224*224 image, ResNet-152 and gtx 1080 8GB gpu. This highly depends on the CNN structure and your GPU (or main) memory. Note if you want to change image size, you probably need to change all the 224*224 number in the code.

2. Run ensemble.py:

   python ensemble.py
   
In ensemble.py, the program first load business features from previous step, then run  xgboost and svm model seperately on the features, finally try to linearly combine them with best coefficient obtained from cross validation.

It should be pointed out here that for xgboost and svm, the hyper-parameters they use also need carefully cross-validation. This step actually takes my long time. For the code readbility, I don't put this step here. The parameters I use here is from cross-validation and not some arbitary settings.

It is also worth noting that SVM actually performs much better on the PCA transformed version of train data. This makes sense since pool5 layer from ResNet is 2048D and contain lots of redundant information. By reducing input data dimension, SVM could gain from regularization effect from PCA. I tried several values and find out 200 is a pretty good value for PCA.

However, Xgboost seems has little improvment or even worse performance on the PCA projected data. This is interesting and actually shows that tree-based Xgboost has some feature selection cabability. PCA projection will certainly lose some information presented to Xgboost so we can just leave Xgboost alone and let it choose features by itself.

optional arguments:

  -h, --help            show this help message and exit
  
  --svm_pca_n SVM_PCA_N
                        the number of pca component you want to project to reduce dimension of features for svm. Default is 200 and it gives satisfactory results. It should be smaller than the CNN extracted feature dimension, 2048 in the case of 'pool5' layer of ResNet-152.
 
  --coef_n COEF_N       how many coefficients do you want to search for the linear combination of xgb and svm. Deafult is 10 and will search 0, 0.1, ..., 1.0. The larger the coef_n is, the finer the search will be. But you also need to wait for longer time.

  --threshold THRESHOLD
                        the probability threshold to determine if label is 1. Default is 0.5.
                        
  --n_fold N_FOLD       the number of folds to cross-validate. Default is 10.
  
  3. Finally, submit the submission_res5c_xgb_svm_ensem.csv file to kaggle to get your score.
