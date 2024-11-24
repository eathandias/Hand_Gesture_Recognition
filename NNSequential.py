import torch
from torch import nn 
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer 
import cv2 as cv2
from torch.autograd import Variable
import imutils
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
path = Path("data/")
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(20000, 64),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5), 
            torch.nn.Linear(64, 8)
        )
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out  
data_path = Path("model.pth")
image_path = path / "Gesture"
train_dir = image_path / "train"
train_data = datasets.ImageFolder(root=train_dir,
                                  target_transform=None)
class_names = train_data.classes
model =  GestureRecognitionModel().to(device)
model.load_state_dict(torch.load(data_path))
model.eval()


cap = cv2.VideoCapture(0)
ok, frame = cap.read()
bg = frame.copy()
kernel = np.ones((3,3),np.uint8)
# Display positions (pixel coordinates)
positions = {
    'hand_pose': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
    'null_pos': (200, 200) # used as null point for mouse control
}
# Tracking
# Bounding box -> (TopRightX, TopRightY, Width, Height)
bbox_initial = (116, 116, 216, 216) # Starting position for bounding box
bbox = bbox_initial
# Capture, process, display loop    
while True:
    # Read a new frame
    ok, frame = cap.read()
    display = frame.copy()
    data_display = np.zeros_like(display, dtype=np.uint8) # Black screen to display data
    if not ok:
        break
    # Start timer
    timer = cv2.getTickCount()
    # Processing
    # First find the absolute difference between the two images
    diff = cv2.absdiff(bg, frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)
    hand_crop = img_dilation[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    # Resize cropped hand and make prediction on gesture
    hand_crop_resized = transforms.ToTensor()(cv2.resize(hand_crop, (100, 100))).unsqueeze(0).to(device)
    prediction = model(hand_crop_resized)
    predi = prediction[0].argmax() # Get the index of the greatest confidence
    gesture = class_names[predi]    
    for i, pred in enumerate(prediction[0]):
        # Draw confidence bar for each gesture
        barx = positions['hand_pose'][0]
        bary = 60 + i*60
        bar_height = 10
        bar_length = int(400 * pred) + barx # calculate length of confidence bar    
        # Make the most confidence prediction green
        if i == predi:
            colour = (0, 255, 0)
        else:
            colour = (0, 0, 255)   
        cv2.putText(data_display, "{}: {}".format(class_names[i],predi), (positions['hand_pose'][0], 20 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), colour, -1, 1)    
    cv2.putText(display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(display, "FPS : " + str(int(fps)), positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
    # Display result
    cv2.imshow("display", display)
    # Display result
    cv2.imshow("data", data_display)
    # Display diff
    cv2.imshow("diff", diff)
    # Display thresh
    cv2.imshow("thresh", thresh)
    # Display mask
    cv2.imshow("img_dilation", img_dilation)
    try:
        # Display hand_crop
        cv2.imshow("hand_crop", hand_crop)
    except:
        pass
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()