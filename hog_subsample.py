import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from utils import *
import json
from utils import non_max_suppression_fast
from scipy.ndimage.measurements import label

def convert_to_color_space(image, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: converted_image = np.copy(image)

    return converted_image



# Define a single function that can extract features using hog sub-sampling and make predictions
def detect_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, 
    spatial_feat=True, hist_feat=True, hog_feat=True):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    
    ctrans_tosearch = convert_to_color_space(img_tosearch, cspace)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            all_features = []
            if spatial_feat == True:
                # print('m num of spatial features', len(spatial_features))
                all_features.append(spatial_features)
            if hist_feat == True:
                # print('m num of hist features', len(hist_features))
                all_features.append(hist_features)
            if hog_feat == True:
                # print('m num of hog features', len(hog_features))
                all_features.append(hog_features)

            all_features = np.concatenate(all_features)
            # Scale features and make a prediction
            # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) 
            test_features = X_scaler.transform([all_features]) 
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append((xbox_left, ytop_draw+ystart, xbox_left+win_draw, ytop_draw+win_draw+ystart))

    return boxes


    # pick = non_max_suppression_fast(np.array(boxes), 0.3)
    # for (startX, startY, endX, endY) in pick:
    #     cv2.rectangle(draw_img, (startX, startY), (endX, endY), (0, 0, 255), 6)

    # return draw_img



if __name__ == '__main__':

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

    model_path = 'model.pkl'
    X_Scaler_path = 'x_scaler.pkl'
    with open(model_path, mode='rb') as fp:
        svm_model = pickle.load(fp)

    with open(X_Scaler_path, mode='rb') as fp:
        X_scaler = pickle.load(fp)

    image_path = './test_images/test1.jpg'
    img = mpimg.imread(image_path)

    ystart = 400
    ystop = 656
    scale = 1.5
        
    boxes = detect_cars(img, ystart, ystop, scale, svm_model, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,boxes)    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)

    plt.imshow(draw_img)
    plt.show()