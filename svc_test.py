# -*- coding:utf-8 -*-

import pickle
import matplotlib.image as mpimg
from utils import object_detect, non_max_suppression_fast, add_heat, apply_threshold, draw_labeled_bboxes
import numpy as np
import cv2
import time
import json
from moviepy.editor import VideoFileClip
from hog_subsample import detect_cars
from scipy.ndimage.measurements import label


model_path = 'model.pkl'
X_Scaler_path = 'x_scaler.pkl'
with open(model_path, mode='rb') as fp:
    svm_model = pickle.load(fp)

with open(X_Scaler_path, mode='rb') as fp:
    X_scaler = pickle.load(fp)

with open('config.json') as data_file:    
    data = json.load(data_file)
    color_space = data["color_space"]

    temp_spatial_size = data["spatial_size"]
    spatial_size = (temp_spatial_size[0], temp_spatial_size[1])    
    
    hist_bins = data["hist_bins"]
    
    pix_per_cell = data["pix_per_cell"]
    cell_per_block = data["cell_per_block"]
    orient = data["orient"]
    hog_channel = data["hog_channel"]

    temp_window_dim = data["window_dim"]
    window_dim = (temp_window_dim[0], temp_window_dim[1])
    winStep = data["winStep"]
    minProb = data["minProb"]
    pyramidScale = data["pyramidScale"]

    ystart = data["ystart"]
    ystop = data["ystop"]

    if data["spatial_feat"] == 1:
        spatial_feat = True
    else:
        spatial_feat = False
    
    if data["hist_feat"] == 1:
        hist_feat = True
    else:
        hist_feat = False
    
    if data["hog_feat"] == 1:
        hog_feat = True
    else:
        hog_feat = False

    print("color space: {}".format(color_space))
    
    print("is spatial_feat:{}".format(spatial_feat))
    print("\tspatial_size:{}".format(spatial_size))
    
    print("is hist_feat:{}".format(hist_feat))
    print("\thist_bins:{}".format(hist_bins))

    print("is hog_feat:{}".format(hog_feat))
    print("\tpix_per_cel:{}".format(pix_per_cell))
    print("\tcell_per_block:{}".format(cell_per_block))
    print("\torient:{}".format(orient)) 


def test_v1():
    image_path = './test_images/test1.jpg'
    
    image = mpimg.imread(image_path)
    image = image.astype(np.float32)/255
    half_image = image[400:680, :]

    t1 = time.time()
    boxes, probs = object_detect(image=half_image, svm_model=svm_model, X_scaler=X_scaler, 
        winDim=window_dim, winStep=winStep, pyramidScale=pyramidScale, minProb=minProb, 
        cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, 
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    pick = non_max_suppression_fast(np.array(boxes), 0.3)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY+400), (endX, endY+400), (0, 0, 255), 2)
    t2 = time.time()
    print(round(t2-t1, 2), 'Seconds to SVC predict...')

    cv2.imshow("result", image)
    cv2.waitKey(0)

def test():
    image_path = './test_images/test4.jpg'
    image = mpimg.imread(image_path)
    result = process_image(image)
    cv2.imshow("result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)



def process_image_v1(image):
    bottom_image = image[400:680, :].astype(np.float32)/255

    boxes, probs = object_detect(image=bottom_image, svm_model=svm_model, X_scaler=X_scaler, 
        winDim=window_dim, winStep=winStep, pyramidScale=pyramidScale, minProb=minProb, 
        cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, 
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    pick = non_max_suppression_fast(np.array(boxes), 0.3)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY+400), (endX, endY+400), (0, 0, 255), 2)

    return image

def process_image(image):

    boxes = detect_cars(img=image, 
        ystart=ystart, ystop=ystop, scale=pyramidScale, 
        svc=svm_model, X_scaler=X_scaler, cspace=color_space, 
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
        spatial_size=spatial_size, 
        hist_bins=hist_bins,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    # pick = non_max_suppression_fast(np.array(boxes), 0.3)
    draw_img = np.copy(image)
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(draw_img, (startX, startY), (endX, endY), (0, 0, 255), 6)

    # heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # # Add heat to each box in box list
    # heat = add_heat(heat,boxes)    
    # # Apply threshold to help remove false positives
    # heat = apply_threshold(heat,1)
    # # Visualize the heatmap when displaying    
    # heatmap = np.clip(heat, 0, 255)
    # # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    # draw_img = draw_labeled_bboxes(np.copy(image), labels)


    return draw_img

# test_video
# project_video
VIDEO_FLAG = False
TEST_VIDEO = True
if VIDEO_FLAG:
    if TEST_VIDEO == True:
        project_video_output = 'test_video_output.mp4'
        clip1 = VideoFileClip("test_video.mp4")
    else:
        project_video_output = 'project_video_output.mp4'
        clip1 = VideoFileClip("project_video.mp4")

    # clip1 = VideoFileClip("test_video.mp4").subclip(3, 11)
    proj_clip = clip1.fl_image(process_image)
    proj_clip.write_videofile(project_video_output, audio=False)
else:
    test()
