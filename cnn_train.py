# -*- coding:utf-8 -*-
# import os
import glob
import os
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten, Input, Conv2D, MaxPooling2D, Lambda
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt


'''
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
'''

def create_model(input_shape=(64,64,3)):
    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    # 1st conv layer with 128 filter, 3x3 each, 50% dropout
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))  
    model.add(Dropout(0.5))
    # 2nd conv layer with 128 filter, 3x3 each, 50% dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(Dropout(0.5))
    # 3rd conv layer with 128 filter, 3x3 each, 8x8 pooling and dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8), name='maxpool'))
    model.add(Dropout(0.5))
    # This acts like a 128 neuron dense layer
    model.add(Convolution2D(128, 8, 8,activation="relu",name="dense1")) 
    model.add(Dropout(0.5))
    # This is like a 1 neuron dense layer with tanh [-1, 1]
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) 
    
    return model


# Plot the results of the training
def plot_results(history):
    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def search_cars(img):
    # We crop the image to 440-660px in the vertical direction
    cropped = img[400:660, 0:1280]
    heat = heatmodel.predict(cropped.reshape(1,cropped.shape[0],cropped.shape[1],cropped.shape[2]))
    # This finds us rectangles that are interesting
    xx, yy = np.meshgrid(np.arange(heat.shape[2]),np.arange(heat.shape[1]))
    x = (xx[heat[0,:,:,0]>0.9999999])
    y = (yy[heat[0,:,:,0]>0.9999999])
    hot_windows = []
    # We save those rects in a list
    for i,j in zip(x,y):
        hot_windows.append(((i*8,400 + j*8), (i*8+64,400 +j*8+64)))
    return hot_windows

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img

car_image_path = '../../Dataset/cars_notcars_dataset/vehicles'
none_car_image_path = '../../Dataset/cars_notcars_dataset/non-vehicles'

car_files = glob.glob(car_image_path + "/*/*.png")
notcar_files = glob.glob(none_car_image_path + "/*/*.png")
# car_files = []
# notcar_files = []
# # Get all image files of car
# for root, dirs, files in os.walk(car_image_path):
#     car_files.extend([os.path.join(root, file) for file in files])
# # Filter out None image files
# matchings = [s for s in car_files if "DS_Store" in s]
# if len(matchings) > 0:
#     for matching in matchings:
#         car_files.remove(matching)
# # Get all image files of none car 
# for root, dirs, files in os.walk(none_car_image_path):
#     notcar_files.extend([os.path.join(root, file) for file in files])
# # Filter out None image files
# matchings = [s for s in notcar_files if "DS_Store" in s]
# if len(matchings) > 0:
#     for matching in matchings:
#         notcar_files.remove(matching)


X = []
for image_file in car_files:
    image = cv2.imread(image_file)
    X.append(image)
    # X.append(skimage.io.imread(image_file))
for image_file in notcar_files:
    image = cv2.imread(image_file)
    X.append(image)
    # X.append(skimage.io.imread(image_file))

X = np.array(X)
print("num of x:{}".format(X.shape))
y = np.hstack( (np.ones(len(car_files)), np.zeros(len(notcar_files))) )
print("number of y:{}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



TRAIN_FLAG = True
if TRAIN_FLAG == True:

    model = create_model()
    model.add(Flatten())
    model.summary()

    # filepath = "weights.hdf5"
    filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint]

    # rmsprop, adam
    # loss='binary_crossentropy',
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=512, nb_epoch=20, verbose=2, callbacks=callbacks_list, validation_data=(X_test, y_test))
    model.save_weights('cnn_model.h5')

    plot_results(history)

else:
    
    # Init a version of our network with another resolution without the flatten layer
    heatmodel = create_model((260, 1280, 3))
    # Load the weights
    heatmodel.load_weights('cnn_model.h5')

    # Search for our windows
    image_path = './test_images/test1.jpg'
    image = cv2.imread(image_path)
    hot_windows = search_cars(image)
    
    # # Draw the found boxes on the test image
    # window_img = draw_boxes(image, hot_windows, (0, 255, 0), 6)                    

    # Create image for the heat similar to one shown above 
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 3)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    boxes = label(heatmap)

    # Create the final image
    draw_img = draw_labeled_bboxes(np.copy(image), boxes)

    cv2.imshow('test', draw_img)
    cv2.waitKey(0)


