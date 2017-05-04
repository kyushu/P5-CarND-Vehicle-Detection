# -*- coding:utf-8 -*-
import os
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils import *
import json

with open('config.json') as data_file:    
    data = json.load(data_file)
    color_space = data["color_space"]
    pix_per_cell = data["pix_per_cell"]
    cell_per_block = data["cell_per_block"]
    orient = data["orient"]
    temp_spatial_size = data["spatial_size"]
    spatial_size = (temp_spatial_size[0], temp_spatial_size[1])
    hist_bins = data["hist_bins"]
    hog_channel = data["hog_channel"]

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


# 1. load features
db_path = 'features.hdf5'
model_path = 'model.pkl'
X_Scaler_path = 'x_scaler.pkl'
if os.path.isfile(db_path):
    print("features hdf5 file is exist")

    scaled_x, y = load_dataset(db_path, "features")

else:
    # Here we use Positive-Negative Learning
    # So we prepare two dataset set for CAR(Positive) and NOT-CAR(Negative)

    print("features hdf5 file is not exist")
    print("[INFO] Extracting Features...")
    t1 = time.time()

    car_image_path = '../../Dataset/cars_notcars_dataset/vehicles'
    none_car_image_path = '../../Dataset/cars_notcars_dataset/non-vehicles'

    car_files = []
    notcar_files = []

    # Get all image files of car
    for root, dirs, files in os.walk(car_image_path):
        car_files.extend([os.path.join(root, file) for file in files])
    # Filter out None image files
    matchings = [s for s in car_files if "DS_Store" in s]
    if len(matchings) > 0:
        for matching in matchings:
            car_files.remove(matching)
    # Get all image files of none car 
    for root, dirs, files in os.walk(none_car_image_path):
        notcar_files.extend([os.path.join(root, file) for file in files])
    # Filter out None image files
    matchings = [s for s in notcar_files if "DS_Store" in s]
    if len(matchings) > 0:
        for matching in matchings:
            notcar_files.remove(matching)

    car_features = extract_features(car_files, 
        color_space=color_space,
        spatial_size=spatial_size, 
        hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, 
        spatial_feat=spatial_feat, 
        hist_feat=hist_feat, 
        hog_feat=hog_feat)
    notcar_features = extract_features(notcar_files, 
        color_space=color_space,
        spatial_size=spatial_size, 
        hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, 
        spatial_feat=spatial_feat, 
        hist_feat=hist_feat, 
        hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_x = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack( (np.ones(len(car_features)), np.zeros(len(notcar_features))) )

    t2 = time.time()
    print(round(t2-t1, 2), 'Seconds to extract Features...')

    # Dump Scaled feature and label to hdf5 file
    dump_dataset(scaled_x, y, db_path, "features")

    pickle.dump( X_scaler, open( X_Scaler_path, "wb" ) )


# Split up data into randomized training and test sets
TRAIN_FLAG = True
if TRAIN_FLAG:
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=rand_state)
    svc_model = SVC(kernel="linear", C=0.01, probability=True, random_state=42)

    print("[INFO] Training classifier...")
    t1 = time.time()
    svc_model.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t1, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc_model.score(X_test, y_test), 4))

    print("[INFO] Dumping classifier...")
    pickle.dump( svc_model, open( model_path, "wb" ) )

