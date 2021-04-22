import torch
import torch.nn as nn
import numpy as np
import math
import cv2

def get_final_coords(batch_heatmaps):
    batch_heatmaps = batch_heatmaps.data.cpu().numpy()
    sample_num, joints_num, h, w = batch_heatmaps.shape
    coords = get_max_coords(batch_heatmaps).astype('float32')
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][1]+0.5))
            py = int(math.floor(coords[n][p][0]+0.5))
            if 1 < px < w-1 and 1<py<h-1:
                diff = np.array([
                    hm[py+1][px]-hm[py-1][px],
                    hm[py][px+1]-hm[py][px-1]
                ])
                coords[n][p] += np.sign(diff) * 0.25
    return(coords)

def get_max_coords(batch_heatmaps):
    sample_num, joints_num, h, w = batch_heatmaps.shape
    y = batch_heatmaps.reshape(sample_num, joints_num, -1)
    idx = np.argmax(y, axis=-1)
    coords = np.stack(np.divmod(idx, w), axis=-1)
    return coords

def mr_detect(video, mr_model, use_cuda=True, threshold=0.5):
    # padding
    if len(video)%32>0:
        s = len(video)%32
        video = np.pad(video, (0, 32-s, (0,0), (0,0), (0,0)), mode='constant')
    
    video = video / 255

    # model predict
    pred_mask, pred_cls, pred_kf = [], [], []
    clip_num = math.ceil(len(video)/32)
    l = len(video) % 32
    l = 32 if l==0 else l
    mr_model.eval()
    with torch.no_grad():
        for i in range(clip_num):
            input_tensor = torch.from_numpy(video[i*32:(i+1)*32]).permute(0,3,1,2).contiguous()[None].float()
            if use_cuda:
                input_tensor = input_tensor.cuda()
            output = mr_model(input_tensor)
            pred_mask.append(output[0].data.cpu().numpy()[0])
            pred_cls.append(output[1].data.cpu().numpy()[0])
            pred_kf.append(output[2].data.cpu().numpy()[0])
    
    pred_cls = np.concatenate(pred_cls)
    pred_mask = np.concatenate(pred_mask)
    pred_kf = np.concatenate(pred_kf)

    score = np.max(pred_cls, axis=0)[1]
    kf = np.argmax(pred_kf[:,0])
    pred_mask = (pred_mask>0.5).astype('uint8')

    # quantification
    if score > threshold:
        jet_mask = np.ascontiguousarray(pred_mask[kf,0])
        conts = cv2.findContours(jet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        jet_cont = conts[np.argmax(np.asarray(list(map(len, conts))))]
        jet_cont = np.array(jet_cont, dtype='int32').reshape((len(jet_cont),2))
        jet_region = cv2.contourArea(jet_cont)
        jet_area = np.sum(jet_region)
        la_region = np.ascontiguousarray(pred_mask[kf,2], dtype='uint8')
        la_area = np.sum(la_region|jet_region)
        return score, kf, pred_mask[kf], jet_area/la_area 

def ar_detect(video, ar_model, lvot_model, use_cuda=True, threshold=0.5):
    # padding
    if len(video)%32>0:
        s = len(video)%32
        video = np.pad(video, (0, 32-s, (0,0), (0,0), (0,0)), mode='constant')
    
    video = video / 255

    # model predict
    pred_mask, pred_cls, pred_kf = [], [], []
    clip_num = math.ceil(len(video)/32)
    l = len(video) % 32
    l = 32 if l==0 else l
    ar_model.eval()
    with torch.no_grad():
        for i in range(clip_num):
            input_tensor = torch.from_numpy(video[i*32:(i+1)*32]).permute(0,3,1,2).contiguous()[None].float()
            if use_cuda:
                input_tensor = input_tensor.cuda()
            output = ar_model(input_tensor)
            pred_mask.append(output[0].data.cpu().numpy()[0])
            pred_cls.append(output[1].data.cpu().numpy()[0])
            pred_kf.append(output[2].data.cpu().numpy()[0])
    
    pred_cls = np.concatenate(pred_cls)
    pred_mask = np.concatenate(pred_mask)
    pred_kf = np.concatenate(pred_kf)

    score = np.max(pred_cls, axis=0)[1]
    kf = np.argmax(pred_kf[:,0])
    pred_mask = (pred_mask>0.5).astype('uint8')

    # quantification
    if score > threshold:
        jet_mask = np.ascontiguousarray(pred_mask[kf,0])
        conts = cv2.findContours(jet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        jet_cont = conts[np.argmax(np.asarray(list(map(len, conts))))]
        jet_cont = np.array(jet_cont, dtype='int32').reshape((len(jet_cont),2))

        # get lvot endpoints
        input_tensor = torch.from_numpy(video[kf]).permute(2,0,1).contiguous()[None].float()
        if use_cuda:
            kp_heatmap = lvot_model(input_tensor)
            coords = get_final_coords(kp_heatmap)[0]

        # get jet width and lvot diameter
        bvec = coords[0] - coords[1]
        nvec = np.copy(bvec)
        nvec[[0,1]] = nvec[[1,0]]
        nvec[0] = -nvec[0]
        projection = np.abs(np.sum((jet_cont-coords[0])*nvec, axis=-1)/np.linalg.norm(nvec))
        points = jet_cont[projection<=1]
        pj = np.sum((points-coords[0])*bvec, axis=-1)
        jet_sct1_idx, jet_sct2_idx = np.argmin(pj), np.argmax(pj)
        jet_sct1, jet_sct2 = points[jet_sct1_idx], points[jet_sct2_idx]
        jet_width = np.linalg.norm(jet_sct2-jet_sct1)
        lvot_width = np.linalg.norm(coords[0]-coords[1])
        return score, kf, pred_mask[kf], jet_width/lvot_width







