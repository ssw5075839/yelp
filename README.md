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

In CNN feature extraction, we only forward CNN up to certain layer rather than all the way to the top because lower layer usually contains much more abstract information. For example, in ResNet-152, we forward CNN up to ‘pool5’ layer. This layer has dimension of 4096 and just below the last fc1000 layer. We believe this layer contains enough abstract information while that information is not too low-level to be useful.

Then for each business, all those 4096D features from each of its photos are aggregated to form the feature of this particular business. However, this aggregation process is not trivial. One simple method is just taking the average of all the 4096D features. Another fancier method, as in the 1st place solution http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/ indicated, is computing fisher vector for these 4096D features. However, for those fancy methods, one limiting factor is the number of photos for each business varies a lot, from 2 to 2000. For example, to compute fisher vector, how many components should we use in GMM? 2 may be too small since we have 9 labels here. 9 seem reasonable but a lot of businesses contain less than 9 photos and those precious training data have to be obsoleted.

Actually, the 1st place solution doesn’t show too much solid advantage over others. I would personally say its leading in score is due to its ensemble of many different CNNs. (0.8317 vs 0.8241)

After so many trail and errors, I discovered that a simple average is the most efficient way to aggregate the 4096D features and decide to use it unless others tell me other efficient way.

After business-level features are computed, we can use our favorite classifiers to deal with this problem. I use Xgboost and SVM to classify the 9 labels. One detail is that I find SVM may need PCA the 4096D features to 256D first. In this way I can achieve a little better score. Note that Xgboost doesn’t need PCA first because it has somewhat automatic feature selection cabability.

Another point is that we cannot train the CNN end to end because at the top the classifier (Xgboost) cannot back-propgated. Maybe later on we can incorporate classifier capable of back-propgate to the caffe framework and dealing with average of various number of input. If this goal is achieved, the fine tuned CNN is expected to achieve better performance

Detail code running instruction:

Pre-request:

Python (Recommand Anaconda https://www.continuum.io/downloads)
caffe (http://caffe.berkeleyvision.org/)
Resnet (https://github.com/KaimingHe/deep-residual-networks)
Competetion data (https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)

1.
