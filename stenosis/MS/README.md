# Mitral Stenosis(MS) 
The diagnosis of MS is divided into two steps:
step1: the classification of MS, 
step2(if the result of step1 is positive): segmentation, key points detection and calculation of mitral valve area(mva)

## classification of MS
### training
The training and validation data should be preprocessed into npy file. Every piece of data includes one or two view video in a folder. Data list should be a list of (npy_folder, label), in the format of pickle file. Fill the 'train_path' and 'valid_path' in the 'train_config.yaml' with the path of training data list and validation data list respectively.
training example:
    ~~~shell
    python3 classification/train.py -c classification/train_config_example.yaml
    ~~~
### predicting
fill 'model_path' in the 'predict_classification.py' with trained model 
    ~~~shell
    python3 predict_classification.py
    ~~~

## key points detection and calculation of mva
### training of dectection model
First, segmentation of the CW image. Fill the 'train_path', 'valid_path' and 'data_root' in the train_config file with training, validation datalist and data root path.
    ~~~shell
    python3 segmentation/train.py -c segmentation/train_config_example.yaml
    ~~~
Second, dectection of the key points. Fill the 'train_path' and 'valid_path' in the train_config file.
    ~~~shell
    python3 key_point_dectection/train.py -c key_point_dectection/train_config_example.yaml
    ~~~
### predicting
Fill the 'segment_model_path' and 'points_model_path' in 'predic_calculate_mva.py' with trained model.
    ~~~shell
    python3 predic_calculate_mva.py
    ~~~
