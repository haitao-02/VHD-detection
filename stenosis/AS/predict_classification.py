from classification import ASClsInfer
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'

model_path = ''
infer = ASClsInfer(model_path)
infer.load_model()
res = infer('example_data/classification')
print(res)