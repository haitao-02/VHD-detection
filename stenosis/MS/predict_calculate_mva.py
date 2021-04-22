import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
from PIL import Image
from segmentation import MSCWInfer
from key_point_detection import CWPoints_infer
import cv2

segment_model_path = ''
segmentor = MSCWInfer(segment_model_path)
segmentor.load_model()

points_model_path = ''
points_infer = CWPoints_infer(points_model_path)
points_infer.load_model()

# segment cw image
img = Image.open('example_data/calculate_cw/A4C-MV-CW.png')
img = np.array(img)
mask = segmentor(img)
# print(mask.shape)
# save mask
# mask_ = Image.fromarray(mask)
# mask_.save('example_data/calculate_cw/A4C-MV-CW_mask.png', quality=95)

mask = Image.open('example_data/calculate_cw/A4C-MV-CW_mask.png')
mask = np.array(mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# data from dicom metadata
xmin = 120
xmax = 724
mid_line = 226
ymax = 580
ymin = 227
dx = 0.00449994375
dy = -2.61704033131
# remove bad contours
y_maxs = []
for cn in contours:
    ys = cn[:, :, 1]
    y_maxs.append(max(ys))
rm_ids = []
for idx, ym in enumerate(y_maxs):
    if (mid_line+ymin)-ym > 0.1*mid_line:
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

wids = []
for cn in contours:
    xs = cn[:, :, 0]
    wids.append(max(xs)-min(xs))
rm_idx = []
for idx, w in enumerate(wids):
    if w < 0.5*max(wids):
        rm_idx.append(idx)
contours = [contours[i] for i in range(len(contours)) if i not in rm_idx]
# get waves
up_lefts = []
box_sizes = []
boxes = []
for con in contours:
    con = np.squeeze(con)
    ul = np.min(con, axis=0).astype(np.int32)
    dr = np.max(con, axis=0).astype(np.int32)
    wh = dr-ul+1
    center = (ul + wh/2).astype(np.int32)
    boxsize = int(np.max(wh)*1.2)
    x1y1 = center - boxsize//2
    x1, y1 = x1y1
    x2, y2 = x1y1 + boxsize
    if x1<0 or y1<0 or x2>=mask.shape[1] or y2>=mask.shape[0]:
        continue
    imgT = img[y1:y2, x1:x2]
    imgT = cv2.resize(imgT, (256, 256))
    up_lefts.append(x1y1)
    box_sizes.append(boxsize)
    boxes.append(imgT)
# predict key points
locs = []
for b in boxes:
    loc = points_infer(b)
    locs.append(loc)
# calculate mva
mvas = []
for i in range(len(locs)):
    loc= locs[i]
    ul = up_lefts[i]
    size = box_sizes[i]
    xy1 = ul + (loc[0]*size+0.5).astype('int')
    xy2 = ul + (loc[1]*size+0.5).astype('int')
    pts = np.array([xy1, xy2])
    cv2.polylines(img, [pts], 0, (255, 0, 0), 1)
    for p in pts:
        cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
    xs = list(pts[:, 0])
    ys = list(pts[:, 1])
    x = [(n-xmin)*dx for n in xs]
    y = [((ymin+mid_line)-n)*abs(dy) for n in ys]
    x = np.array(x)
    y = np.array(y)
    data = np.column_stack((x, y))
    p1 = data[0]
    p2 = data[1]
    k = (p1[1]-p2[1])/(p2[0]-p1[0])
    t = (1-0.5*np.sqrt(2))*p1[1]/k
    mva = 220/(t*1000)
    mvas.append(mva)
# save predict points image
# img_ = Image.fromarray(img)
# img_.save('example_data/calculate_cw/A4C-MV-CW_predict_points.png', quality=95)
mva = np.mean(mvas)
print('mva:', mva)
