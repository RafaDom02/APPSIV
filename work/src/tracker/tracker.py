import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment # Added for Hungarian algorithm

import motmetrics as mm # Keep for potential later use in evaluation scripts
from torchvision.ops.boxes import clip_boxes_to_image, nms # nms might be useful for track candidates


def calculate_iou_matrix(boxes1, boxes2):
	"""
	Calculates IoU matrix between two sets of boxes.
	boxes1: (N, 4) tensor of N boxes [x1, y1, x2, y2]
	boxes2: (M, 4) tensor of M boxes [x1, y1, x2, y2]
	Returns: (N, M) IoU matrix
	"""
	if boxes1.numel() == 0 or boxes2.numel() == 0:
		return torch.empty((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device if boxes1.numel() > 0 else boxes2.device)

	boxes1_expanded = boxes1.unsqueeze(1) # N, 1, 4
	boxes2_expanded = boxes2.unsqueeze(0) # 1, M, 4

	xi1 = torch.max(boxes1_expanded[..., 0], boxes2_expanded[..., 0])
	yi1 = torch.max(boxes1_expanded[..., 1], boxes2_expanded[..., 1])
	xi2 = torch.min(boxes1_expanded[..., 2], boxes2_expanded[..., 2])
	yi2 = torch.min(boxes1_expanded[..., 3], boxes2_expanded[..., 3])

	inter_width = torch.clamp(xi2 - xi1, min=0)
	inter_height = torch.clamp(yi2 - yi1, min=0)
	inter_area = inter_width * inter_height

	area1 = (boxes1_expanded[..., 2] - boxes1_expanded[..., 0]) * (boxes1_expanded[..., 3] - boxes1_expanded[..., 1])
	area2 = (boxes2_expanded[..., 2] - boxes2_expanded[..., 0]) * (boxes2_expanded[..., 3] - boxes2_expanded[..., 1])
	union_area = area1 + area2 - inter_area

	iou = inter_area / torch.clamp(union_area, min=1e-6)
	return iou


class Track(object):
	"""This class contains all necessary information for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box  # Current bounding box [x1, y1, x2, y2]
		self.score = score # Current confidence score
		
		self.hits = 1  # Number of consecutive frames this track has been matched
		self.total_hits = 1 # Total number of matches over its lifetime
		self.age = 0 # Number of frames this track has existed (increments each step)
		self.time_since_update = 0 # Number of frames since last successful match
		self.confirmed = False # Track status

	def update(self, box, score):
		"""Updates the track with a new detection."""
		self.box = box
		self.score = score
		self.hits += 1
		self.total_hits +=1
		self.time_since_update = 0

	def predict(self):
		"""Predicts the next state of the track."""
		return self.box

	def increment_age_and_staleness(self):
		"""Increments age and time_since_update. Resets consecutive hits if not updated."""
		self.age += 1
		self.time_since_update += 1
		if self.time_since_update > 0: # If not updated in the current step
			self.hits = 0 # Reset consecutive hits

	def mark_confirmed(self):
		"""Marks the track as confirmed."""
		self.confirmed = True
		
	def is_lost(self, max_frames_missed):
		"""Checks if the track is considered lost."""
		return self.time_since_update > max_frames_missed


class Tracker:
	"""The main tracking class. Implements tracking-by-detection with IoU matching."""

	def __init__(self, obj_detect, min_confidence_score=0.5, iou_match_threshold=0.3, 
				 max_frames_missed=30, min_acceptance_hits=3):
		self.obj_detect = obj_detect
		self.min_confidence_score = min_confidence_score
		self.iou_match_threshold = iou_match_threshold
		self.max_frames_missed = max_frames_missed
		self.min_acceptance_hits = min_acceptance_hits # Min total_hits to confirm a track

		self.tracks = []
		self.next_track_id = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None # For MOT evaluation

	def reset(self, hard=True):
		"""Resets the tracker state."""
		self.tracks = []
		if hard:
			self.next_track_id = 0
			self.results = {}
			self.im_index = 0

	def get_pos(self):
		"""Get the positions of all confirmed and active (recently seen) tracks."""
		if not self.tracks:
			return torch.empty((0, 4), device='cpu') 
		
		# Consider tracks that are confirmed or have been seen recently
		active_tracks = [t for t in self.tracks if t.confirmed or t.time_since_update <= 1]
		if not active_tracks:
			return torch.empty((0, 4), device='cpu')

		# Ensure boxes are on CPU and stacked correctly
		boxes_list = [t.box.cpu() if isinstance(t.box, torch.Tensor) else torch.tensor(t.box, device='cpu') for t in active_tracks]
		
		if not boxes_list:
			return torch.empty((0,4), device='cpu')
		return torch.stack(boxes_list, 0)


	def _data_association(self, detected_boxes, detected_scores):
		"""Performs data association between existing tracks and new detections."""
		num_tracks = len(self.tracks)
		num_detections = detected_boxes.shape[0]

		matched_track_indices = []
		unmatched_detection_indices = list(range(num_detections))
		
		if num_tracks == 0: # No existing tracks, all detections are new
			for i in unmatched_detection_indices:
				new_track = Track(detected_boxes[i], detected_scores[i], self.next_track_id)
				self.tracks.append(new_track)
				self.next_track_id += 1
			return

		# Predict next state for existing tracks
		track_predicted_boxes = torch.stack([t.predict() for t in self.tracks])

		# Calculate IoU cost matrix (cost = 1 - IoU)
		# Ensure boxes are on the same device for IoU calculation
		device = detected_boxes.device
		iou_matrix = calculate_iou_matrix(track_predicted_boxes.to(device), detected_boxes)
		cost_matrix = (1 - iou_matrix).cpu().numpy() # Hungarian algorithm needs CPU numpy array

		# Hungarian algorithm for assignment
		track_indices, detection_indices = linear_sum_assignment(cost_matrix)

		for track_idx, det_idx in zip(track_indices, detection_indices):
			if iou_matrix[track_idx, det_idx] >= self.iou_match_threshold:
				self.tracks[track_idx].update(detected_boxes[det_idx], detected_scores[det_idx])
				matched_track_indices.append(track_idx)
				if det_idx in unmatched_detection_indices:
					unmatched_detection_indices.remove(det_idx)
		
		# Create new tracks for unmatched detections
		for det_idx in unmatched_detection_indices:
			new_track = Track(detected_boxes[det_idx], detected_scores[det_idx], self.next_track_id)
			self.tracks.append(new_track)
			self.next_track_id += 1
			
		# Note: Unmatched tracks are handled by increment_age_and_staleness and is_lost logic in step()

	def step(self, frame):
		"""Processes a single frame to perform tracking."""
		# self.im_index += 1 # Moved to the end

		# 1. Object Detection
		# Assuming obj_detect.detect returns boxes and scores on CPU as per original object_detector.py
		raw_boxes, raw_scores = self.obj_detect.detect(frame['img']) 

		# 2. Filter detections by confidence
		keep_indices = raw_scores >= self.min_confidence_score
		detected_boxes = raw_boxes[keep_indices]
		detected_scores = raw_scores[keep_indices]

		# 3. Increment age and staleness for all tracks
		for t in self.tracks:
			t.increment_age_and_staleness()

		# 4. Data Association
		self._data_association(detected_boxes, detected_scores)

		# 5. Update track states and manage lifecycle
		active_tracks = []
		for t in self.tracks:
			if not t.confirmed and t.total_hits >= self.min_acceptance_hits:
				t.mark_confirmed()

			if not t.is_lost(self.max_frames_missed):
				active_tracks.append(t)
				# Store results for confirmed tracks
				if t.confirmed:
					if t.id not in self.results:
						self.results[t.id] = {}
					
					box_to_store = t.box.cpu().numpy() if isinstance(t.box, torch.Tensor) else np.array(t.box)
					score_to_store = t.score.cpu().numpy() if isinstance(t.score, torch.Tensor) else np.array([t.score])
					
					# Ensure box_to_store is [x1,y1,x2,y2] and then convert to [x,y,w,h] if needed by results format.
					# Current baseline stored [x1,y1,x2,y2,score]
					self.results[t.id][self.im_index] = np.concatenate([box_to_store, score_to_store.reshape(1)])
			# else: track is lost and will be removed from self.tracks

		self.tracks = active_tracks
		self.im_index += 1 # Incremented at the end of processing the frame

	def get_results(self):
		"""Returns the tracking results."""
		return self.results
