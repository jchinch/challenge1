#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:41:16 2020

@author: jatinchinchkar
"""
import torch
import torchvision 
from torchvision import models
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (255,255,255), (255,255,255), (64, 128, 128), (1,1,1),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

vid = cv2.VideoCapture("./challenge1.mp4")
frames,frame_np = [],[]
count = 0
while True:
    ret,frame = vid.read()
    if ret:
        name = "./frame"+str(count)+".jpg"
        temp = cv2.resize(frame, (640,640),interpolation=cv2.INTER_CUBIC)/255
        frame_np.append(temp)
        f = Image.fromarray(frame).resize((640,640),Image.BILINEAR)
        frames.append(f)
        count += 1
    else:
        break
vid.release()
cv2.destroyAllWindows()

out_vid = []
trf = T.Compose([T.Resize(640),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
for i in range(len(frames)):
    ip = trf(frames[i]).unsqueeze(0)
    op = dlab(ip)['out']
    om = torch.argmax(op.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    rgb = decode_segmap(om)
    output = rgb*frame_np[i]
    out_vid.append(output)
final_img = []
back = np.array(Image.open("./background.jpg").resize((640,640), Image.BILINEAR), np.float64)/255
for i in range(len(frames)):
    temp = out_vid[i].astype(float)
    thr, alpha = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(alpha, (7,7), 0)
    alpha = alpha/255
    fore = cv2.multiply(alpha,temp)
    backg = cv2.multiply((1-alpha),back)
    outp = cv2.add(fore,backg)
    final_img.append(255*outp)


out = cv2.VideoWriter("./output.avi",cv2.VideoWriter_fourcc('M','P','E','G'), 30, (640,640))
for i in range(len(final_img)):
    out.write(np.uint8(final_img[i])) 
out.release()

