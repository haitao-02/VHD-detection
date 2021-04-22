# Aortic Stenosis(AS) 
The diagnosis of AS is divided into two steps:
step1: the classification of AS, 
step2(if the result of step1 is positive): segmentation and calculation of parameters

## classification of AS
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

## segmentation and calculation of parameters
### training of segmentation model
Fill the 'train_path', 'valid_path' and 'data_root' in the train_config file with training, validation datalist and data root path.
    ~~~shell
    python3 segmentation/train.py -c segmentation/train_config_example.yaml
    ~~~
### predicting and calculation
Fill the 'model_path' in 'predic_calculate_params.py' with trained model.
    ~~~shell
    python3 predic_calculate_params.py
    ~~~
