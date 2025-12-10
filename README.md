# Skin-Cancer-Classification-Final-Project
Skin Cancer classification final project
(Our up to date code is in skin_cancer_classif.ipynb)

Status 11/26/25:
  - Deployed model and ran 20 epochs on 80 % training 20% validation split.
  - Graphed validation and training accuracies along with validation and training losses.
  - Determined model is overfitting where our model is starting to memorize some of that training data.
  - Next steps are to implement techniques that address overfitting, one is dropout that we learned in lecture.

Status 12/8/25:
  - Implemented recommended optimizations to tackle overfitting in our CNN model. These optimizations we included were:
      - Data Augmentation
      - Dropout Layer
      - Batch Normalization
      - Weighted / Penalized loss
      - Increasing number of epochs
  - Analysis of original model's result with no optimization compared to results after adding all the optimizations shows our model
    is performing much better, and no behaviour of overfitting or underfitting is present.
