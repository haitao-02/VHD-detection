from unet_rcnn import UnetRCNN
from metrics import Accuracy, Precision, Recall, AvgMeterWrapper, ConfusionMatrix, MetricWrapper, Auc, MultiLabelIoU, MultiLabelAccuracy
from loss import RgLoss
from dataset import RgLoader
__all__ = ['model_dict', 'loader_dict', 'loss_dict', 'metric_dict']

model_dict = {
    'UnetRCNN': UnetRCNN,
}

loader_dict = {
    'RgLoader':RgLoader
}

loss_dict = {
    'RgLoss': RgLoss
}

metric_dict = {
    'cls': {
        'Accuracy':AvgMeterWrapper(Accuracy()),
        'Specificity & Recall': AvgMeterWrapper(Recall()),
        'NPV & Precision':AvgMeterWrapper(Precision()),
        'AUC': MetricWrapper(metric=Auc(), idx=1),
        'Confusion_Matrix': ConfusionMatrix(2)
    },
    'seg': {
        'IoU':AvgMeterWrapper(MultiLabelIoU())
    },
    'kf':{
        'Localization_Accuracy': AvgMeterWrapper(MultiLabelAccuracy())
    },
}