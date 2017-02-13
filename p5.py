import time
import glob
import os
import cv2
import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn import tree
from sklearn.cluster import DBSCAN
from moviepy.editor import *

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to return HOG features and visualization
def get_hog_vectors(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                   visualise=vis, feature_vector=feature_vec)
    return features

# Define a function to extract features from a list of images
def extract_features_fromw(fullimg, windows, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, 
                           hog_channel='ALL', hog_feature_vec=False,
                           x_start_stop=[None, None], y_start_stop=[400, 650],
                           spa_size = 16, n_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    image = fullimg[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1]]
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
                    
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(3):
            hog_features.append(get_hog_vectors(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=hog_feature_vec))        
    else:
        hog_features = get_hog_vectors(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=hog_feature_vec)
    
    n_window = 0
    for window in windows:
        window_image = feature_image[window[0][1]-y_start_stop[0]:window[1][1]-y_start_stop[0], 
                                     window[0][0]-x_start_stop[0]:window[1][0]-x_start_stop[0]]
        # window [[xmin, ymin],[xmax,ymax]]
        hogxmin = int((window[0][0]-x_start_stop[0]) / 8)
        hogymin = int((window[0][1]-y_start_stop[0]) / 8)
        hogxmax = int((window[1][0]-x_start_stop[0]) / 8) - 1
        hogymax = int((window[1][1]-y_start_stop[0]) / 8) - 1
        if hogxmax > hog_features[-1].shape[1]:
            hogxmax = hog_features[-1].shape[1]
            hogxmin = hogxmax-7
        if hogymax > hog_features[-1].shape[0]:
            hogymax = hog_features[-1].shape[0]
            hogymin = hogymax-7
       
        window_features=[]
        if hog_channel == 'ALL':
            for channel in range(3):
                window_features.append(hog_features[channel][hogymin:hogymax, hogxmin:hogxmax].ravel())
            window_features = np.ravel(window_features) 
            
        else:
            window_features.append(hog_features[hogymin:hogymax, hogxmin:hogxmax].ravel())
        
        spatial_features = bin_spatial(window_image, size=(spa_size, spa_size)) 
        hist_features = color_hist(window_image, nbins=n_bins)
        temp_features = np.concatenate([spatial_features, hist_features])
        features.append(np.concatenate([temp_features, window_features]))
    return features

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy


class cluster_on_heatmap():
	def dbscan_cluster(self, heatmap, eps=10, min_samples=1,
					  		 Xmin=48, Xmax=380, Ymin=48, Ymax=250):
		if np.count_nonzero(heatmap) == 0:
			return False, 0, []
		else:
			nzero = np.nonzero(heatmap)
			x = nzero[1]
			y = nzero[0]
			ob = np.array([x,y]).T
			db = DBSCAN(eps=eps, min_samples=min_samples).fit(ob)
			core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
			core_samples_mask[db.core_sample_indices_] = True
			labels = db.labels_
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			boxes = []
			for k in range(n_clusters_):
				class_member_mask = (labels == k)
				xy = ob[class_member_mask & core_samples_mask]
				xmax = np.amax(xy.T[0])
				xmin = np.amin(xy.T[0])
				ymax = np.amax(xy.T[1])
				ymin = np.amin(xy.T[1])
				box = [(xmin, ymin), (xmax, ymax)]
				boxes += [box]
			return True, len(boxes), boxes

# Implements a linear Kalman filter.
class KalmanFilter:
	def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
		self.A = _A                      # State transition matrix.
		self.B = _B                      # Control matrix.
		self.H = _H                      # Observation matrix.
		self.current_state_estimate = _x # Initial state estimate.
		self.current_prob_estimate = _P  # Initial covariance estimate.
		self.Q = _Q                      # Estimated error in process.
		self.R = _R                      # Estimated error in measurements.
	def GetCurrentState(self):
		return self.current_state_estimate
	def Step(self,control_vector,measurement_vector):
		# prediction
		predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_vector
		predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q
		# observation
		innovation = measurement_vector - self.H*predicted_state_estimate
		innovation_covariance = self.H*predicted_prob_estimate*np.transpose(self.H) + self.R
		# update
		kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)
		self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
		size = self.current_prob_estimate.shape[0]
		self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate

class vehicle_tracker():
	def __init__(self):
		self.age = 0
		A = np.matrix([1])
		H = np.matrix([1])
		B = np.matrix([0])
		Q = np.matrix([0.0001])
		R = np.matrix([0.001])
		xhat = np.matrix([0])
		P    = np.matrix([1])
		self.xfilter = KalmanFilter(A,B,H,xhat,P,Q,R)
		A = np.matrix([1])
		H = np.matrix([1])
		B = np.matrix([0])
		Q = np.matrix([0.0001])
		R = np.matrix([0.0001])
		xhat = np.matrix([0])
		P    = np.matrix([1])
		self.yfilter = KalmanFilter(A,B,H,xhat,P,Q,R)
		A = np.matrix([1])
		H = np.matrix([1])
		B = np.matrix([0])
		Q = np.matrix([0.01])
		R = np.matrix([0.1])
		xhat = np.matrix([1])
		P    = np.matrix([1])
		self.wfilter = KalmanFilter(A,B,H,xhat,P,Q,R)
		A = np.matrix([1])
		H = np.matrix([1])
		B = np.matrix([0])
		Q = np.matrix([0.01])
		R = np.matrix([0.1])
		xhat = np.matrix([1])
		P    = np.matrix([1])
		self.hfilter = KalmanFilter(A,B,H,xhat,P,Q,R)

	def update_tracker(self, box, blood):
		# [[xmin, ymin], [xmax, ymax]]
		ux = (box[0][0]+box[1][0])/2.0
		uy = (box[0][1]+box[1][1])/2.0
		uw = box[1][0]-box[0][0]
		uh = box[1][1]-box[0][1]
		self.xfilter.Step(np.matrix([0]),np.matrix([ux]))
		self.yfilter.Step(np.matrix([0]),np.matrix([uy]))
		self.wfilter.Step(np.matrix([0]),np.matrix([uw]))
		self.hfilter.Step(np.matrix([0]),np.matrix([uh]))
		ox = self.xfilter.GetCurrentState()[0][0]
		oy = self.yfilter.GetCurrentState()[0][0]
		ow = self.wfilter.GetCurrentState()[0][0]
		oh = self.hfilter.GetCurrentState()[0][0]
		self.current_bbox = [(int(ox-ow/2), int(oy-oh/2)),(int(ox+ow/2), int(oy+oh/2))]
		if self.age < 10:
			self.age += blood

	def kill_tracker(self, blood):
		self.age = self.age - blood


def full_scan(img,
              cspace=None, 
              orient=None, 
              ppcell=None, 
              cpblck=None, 
              hogchl=None, 
              hog_feature_vec=False,
              x_start_stop=[0, 1280], 
              y_start_stop=[400, 660],
              xy_window=(64, 64), 
              xy_overlap=(0.8, 0.8),
              spsize = None, 
              n_bins=None,
              clf=None,
              scaler=None):
	# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	windows = slide_window( img, 
							x_start_stop=x_start_stop,
							y_start_stop=y_start_stop,
                   			xy_window=xy_window, 
                   			xy_overlap=xy_overlap)
	features = extract_features_fromw(img, windows,
	                                  cspace=cspace, 
	                                  orient=orient, 
	                                  pix_per_cell=ppcell, 
	                                  cell_per_block=cpblck, 
	                                  hog_channel=hogchl, 
	                                  hog_feature_vec=False,
	                                  x_start_stop=x_start_stop, 
	                                  y_start_stop=y_start_stop,
	                                  spa_size = spsize, 
	                                  n_bins=n_bins)
	X = scaler.transform(features)
	prediction = clf.predict(X)

	return windows, prediction

def box_merge(box0, box1):
	axmin = box0[0][0]
	aymin = box0[0][1]
	axmax = box0[1][0]
	aymax = box0[1][1]
	bxmin = box1[0][0]
	bymin = box1[0][1]
	bxmax = box1[1][0]
	bymax = box1[1][1]
	dx = min(axmax, bxmax) - max(axmin, bxmin)
	dy = min(aymax, bymax) - max(aymin, bymin)
	if (dx>=-6) and (dy>=-6):
		mbox = [(min(box0[0][0],box1[0][0]), min(box0[0][1],box1[0][1])), 
				(max(box0[1][0],box1[1][0]), max(box0[1][1],box1[1][1]))]
		return True, mbox
	else:
		return False, None


if __name__ == '__main__':
	with open('classifier/hogtest.pkl', 'rb') as handle:
	    datapacket = pickle.load(handle)
	    clf = datapacket['clf']
	    scaler = datapacket['scaler']
	    cspace = datapacket['setting']['cps']
	    orient = datapacket['setting']['ori']
	    ppcell = datapacket['setting']['ppc']
	    cpblck = datapacket['setting']['cpb']
	    hogchl = datapacket['setting']['hoc']
	    spsize = datapacket['setting']['sps']
	    n_bins = datapacket['setting']['nbs']

	filename = 'project_video.mp4'
	video = VideoFileClip(filename)   #.subclip(39,60)
	clip = []
	current_trackers = []
	Nimg = 0
	try:
		for img in video.iter_frames():
			Nimg += 1
			windows, prediction = full_scan(   img,
						              cspace=cspace, 
						              orient=orient, 
						              ppcell=ppcell, 
						              cpblck=cpblck, 
						              hogchl=hogchl, 
						              hog_feature_vec=False,
						              x_start_stop=[0, 1280], 
						              y_start_stop=[400, 660],
						              xy_window=(64, 64), 
						              xy_overlap=(0.8, 0.8),
						              spsize = spsize, 
						              n_bins=n_bins,
						              clf=clf,
						              scaler=scaler)



			hot_windows = []
			heat = np.zeros_like(img[:,:,0]).astype(np.float)
			for i in range(len(windows)):
			    if prediction[i] == 1:
			        hot_windows.append(windows[i])
			heat = add_heat(heat, hot_windows)   

			heat = apply_threshold(heat, 1)
			cluster, nbox, boxes = cluster_on_heatmap().dbscan_cluster(heat)

			print (boxes)
			if cluster:
				if current_trackers == []:
					for box in boxes:
						# if box[1][0]-box[0][0]>63 and box[1][1]-box[0][1]>63:
						tracker = vehicle_tracker()
						tracker.update_tracker(box, 1)
						current_trackers += [tracker]
				else:
					merge_check = list(range(len(current_trackers)))
					for i in range(len(merge_check)):
						merge_check[i] = 0
					for i in range(nbox):
						for ct in range(len(current_trackers)):
							if boxes[i] is not None:
								merge, mbox = box_merge(boxes[i], current_trackers[ct].current_bbox)
								if merge:
									merge_check[ct] = 1
									boxes[i] = None
									current_trackers[ct].update_tracker(mbox, 2)
					for mc in range(len(merge_check)):
						if merge_check[mc] == 0:
							current_trackers[mc].kill_tracker(3)
					boxes = [b for b in boxes if b is not None]
					print (boxes)
					if len(boxes)>0:
						for box in boxes:
							if box[1][0]-box[0][0]>2 and box[1][1]-box[0][1]>2:
								tracker = vehicle_tracker()
								tracker.update_tracker(box, 3)
								current_trackers += [tracker]

			current_trackers = [ct for ct in current_trackers if ct.age>0]
			if current_trackers != []:
				# merge tracker
				if len(current_trackers)>1:
					tracker_merged = list(range(len(current_trackers)))
					for i in range(len(tracker_merged)):
								tracker_merged[i] = 0
					# check merge
					for ct0 in range(len(current_trackers)):
						for ct1 in range(len(current_trackers)):
							merge, mbox = box_merge(current_trackers[ct0].current_bbox, 
													current_trackers[ct1].current_bbox)
							if merge:
								if current_trackers[ct0].age >= current_trackers[ct1].age:
									current_trackers[ct0].update_tracker(mbox, 2)
									tracker_merged[ct1] += 1
								else:
									current_trackers[ct1].update_tracker(mbox, 2)
									tracker_merged[ct0] += 1
					for tm in range(len(tracker_merged)):
						if tracker_merged[tm] > 0:
							current_trackers[tm].kill_tracker(6)
				for cti in range(len(current_trackers)):
					bbox = current_trackers[cti].current_bbox
					xmin = bbox[0][0]
					ymin = bbox[0][1]
					xmax = bbox[1][0]
					ymax = bbox[1][1]
					if xmax-xmin<64:
						if xmax > 64:
							xmin = xmax-64
						else:
							xmax = xmin+64
					if ymax-ymin<64:
						if ymax > 64:
							ymin = ymax-64
						else:
							ymax = ymin+64
					if xmin > 64:
						xmin = xmin - 32
					if xmax < 1280-64:
						xmax += 32
					if ymin > 64 and ymax < 720-64:
						ymin = ymin-32
						ymax += 32
					xss = [xmin, xmax]
					yss = [ymin, ymax]
					windows, prediction = full_scan(   img,
						              cspace=cspace, 
						              orient=orient, 
						              ppcell=ppcell, 
						              cpblck=cpblck, 
						              hogchl=hogchl, 
						              hog_feature_vec=False,
						              x_start_stop=xss, 
						              y_start_stop=yss,
						              xy_window=(64, 64), 
						              xy_overlap=(0.9, 0.9),
						              spsize = spsize, 
						              n_bins=n_bins,
						              clf=clf,
						              scaler=scaler)
					hot_windows = []
					heat = np.zeros_like(img[:,:,0]).astype(np.float)
					for i in range(len(windows)):
					    if prediction[i] == 1:
					        hot_windows.append(windows[i])
					heat = add_heat(heat, hot_windows)   
					heat = apply_threshold(heat, 0)
					cluster, nbox, boxes = cluster_on_heatmap().dbscan_cluster(heat)
					new_trackers=[]
					if cluster:
						if nbox == 1:
							current_trackers[cti].update_tracker(boxes[0], 2)
						else:
							current_trackers[cti].update_tracker(boxes[0], 0)
							current_trackers[cti].kill_tracker(1)
							new_tracker = vehicle_tracker()
							new_tracker.update_tracker(boxes[-1], 3)
							new_trackers += [new_tracker]
					else:
						current_trackers[cti].kill_tracker(5)
					if new_trackers != []:
						print ('new tracker', new_trackers)
					current_trackers += new_trackers




			current_trackers = [ct for ct in current_trackers if ct.age>0]
			print ('trackers: ', current_trackers)
			cboxes = []
			for ct in current_trackers:
				cboxes += [ct.current_bbox]
				img = draw_boxes(img, cboxes)
				clip += [img]
				print ('valid bbox: ', cboxes)
			cv2.imshow('o', img)
			cv2.waitKey(200)
			print (Nimg, end='\r')
	except KeyboardInterrupt:
		pass
	outvideo = ImageSequenceClip(clip, fps=12)
	outvideo.write_videofile("annotated_project_video.mp4")