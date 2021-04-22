import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
from segmentation import ASCWInfer
from PIL import Image
import numpy as np
import cv2
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

model_path = ''
infer = ASCWInfer(model_path)
infer.load_model()

img = Image.open('example_data/segment_cw/A5C-AV-CW.png')
img = np.array(img)
mask = infer(img)
# print(mask.shape)
# save mask
# mask_ = Image.fromarray(mask)
# mask_.save('example_data/segment_cw/A5C-AV-CW_mask.png', quality=95)

mask = Image.open('example_data/segment_cw/A5C-AV-CW_mask.png')
mask = np.array(mask)
# data from dicom metadata
xmin = 120
xmax = 724
mid_line = 204
ymax = 580
ymin = 227
dx = 0.00449994375
dy = -2.61704033131
# get contour and remove bad contour
contours= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
contours = [c for c in contours if c.shape[0]>100]
y_mins = []
for cn in contours:
    ys = cn[:, :, 1]
    y_mins.append(min(ys))
rm_ids = []
for idx, ym in enumerate(y_mins):
    if ym-mid_line-ymin > 0.1*(ymax-mid_line-ymin):
        rm_ids.append(idx)
contours = [contours[i] for i in range(len(contours)) if i not in rm_ids]
x_minmaxs = []
for cn in contours:
    xs = cn[:, :, 0]
    x_minmaxs.append((min(xs), max(xs)))
rm_idx = []
for idx, xm in enumerate(x_minmaxs):
    if xm[0]-xmin<(xmax-xmin)/20 or xmax-xm[1] <(xmax-xmin)/20:
        rm_idx.append(idx)
contours = [contours[i] for i in range(len(contours)) if i not in rm_idx]

def contour2wave(contour):
    dic_ex = {}
    for point in contour:
        p = point[0]
        if p[0] not in dic_ex:
            dic_ex[p[0]] = p[1]
        else:
            dic_ex[p[0]] = max(dic_ex[p[0]], p[1])
    x = list(dic_ex.keys())
    y = list(dic_ex.values())
    x = [(n-xmin)*dx for n in x]
    y = [(n-ymin-mid_line)*abs(dy) for n in y]
    x = np.array(x)
    y = np.array(y)
    data = np.column_stack((x, y))
    data = list(data)
    data.sort(key = lambda d: d[0])
    data = np.array(data)
    return data
# calculate
results = []
curves = []
for con in contours:
    curve = contour2wave(con)
    x = list(curve[:, 0])
    y = list(curve[:, 1])
    y = np.array(y)
    y_f = savgol_filter(y, 5, 1)
    f = interp1d(x, y_f, 'cubic')
    xx = np.linspace(min(x), max(x), 100)
    yy = f(xx)
    zz = [4*(v/100)**2 for v in yy]
    area = integrate.simps(zz, xx)
    vmax = max(yy)/100
    dpm = area/(max(xx)-min(xx))
    results.append((vmax, dpm))
    curves.append(np.array([xx, yy]))
vs = [r[0] for r in results]
for idx, v in enumerate(vs):
    if v == max(vs):
        tag=idx
choose_curve = curves[tag]
choose_result = results[tag]
choose_contour = contours[tag]
print('V_max:', choose_result[0])
print('delta_Pm:', choose_result[1])

