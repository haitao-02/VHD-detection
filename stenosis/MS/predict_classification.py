from classification import MSClsInfer
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'

model_path = ''
infer = MSClsInfer(model_path='cls.pth')
infer.load_model()
res = infer('example_data/classification/')
print(res)