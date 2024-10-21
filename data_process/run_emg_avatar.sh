#!/bin/bash

# Run the EMG_to_Avatar_model.py script with specified parameters
python EMG_to_Avatar_model.py \
  --data_path "C:/Users/YH006_new/fEMG_to_avatar/data" \
  --ica_flag \
  --save_results \
  --train_deep_learning_model \
  --train_linear_transform \
  --train_enhanced_linear_transform \
  --model_name EnhancedTransformNet \
  --criterion pearson \
  --Early_stopping \
  --segments_length 4 \
  --models LR ETR Ridge Lasso ElasticNet DecisionTreeRegressor RandomForestRegressor

python EMG_to_Avatar_model.py \
  --data_path "C:/Users/YH006_new/fEMG_to_avatar/data" \
  --ica_flag \
  --save_results \
  --train_deep_learning_model \
  --train_linear_transform \
  --train_enhanced_linear_transform \
  --model_name LinearTransformNet \
  --criterion pearson \
  --Early_stopping \
  --segments_length 4 \
  --models LR ETR Ridge Lasso ElasticNet DecisionTreeRegressor RandomForestRegressor

python EMG_to_Avatar_model.py \
  --data_path "C:/Users/YH006_new/fEMG_to_avatar/data" \
  --ica_flag \
  --save_results \
  --train_deep_learning_model \
  --train_linear_transform \
  --train_enhanced_linear_transform \
  --model_name ImprovedEnhancedTransformNet \
  --criterion pearson \
  --Early_stopping \
  --segments_length 4 \
  --models LR ETR Ridge Lasso ElasticNet DecisionTreeRegressor RandomForestRegressor