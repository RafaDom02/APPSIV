import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms

from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect, iou_thresh: float = 0.3, max_misses: int = 5):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None
  
		self.iou_thresh = iou_thresh
		self.max_misses = max_misses
  
	def _add_tracks(self, boxes: torch.Tensor, scores: torch.Tensor):
		"""Crea nuevas pistas a partir de detecciones no asociadas."""
		for i in range(len(boxes)):
			self.tracks.append(
				Track(boxes[i], float(scores[i]), self.track_num)
			)
			self.track_num += 1

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		det_boxes, det_scores = self.obj_detect.detect(frame['img'])
		det_boxes = det_boxes.cpu()
		det_scores = det_scores.cpu()

		# 2) Asociación detección–track
		if len(self.tracks) == 0:
			# Primer frame: arrancamos todas las detecciones
			self._add_tracks(det_boxes, det_scores)

		else:
			# Matriz IoU
			track_boxes = torch.stack([t.box for t in self.tracks])
			iou_mat = box_iou(track_boxes, det_boxes)

			# Hungarian en la matriz de coste (1–IoU)
			cost = 1 - iou_mat.detach().cpu().numpy()
			row_ind, col_ind = linear_sum_assignment(cost)

			matched, unmatched_tracks, unmatched_dets = [], [], []

			# Comprobamos umbral IoU
			for r, c in zip(row_ind, col_ind):
				if iou_mat[r, c] >= self.iou_thresh:
					matched.append((r, c))
				else:
					unmatched_tracks.append(r)
					unmatched_dets.append(c)

			# Tracks no asignados y detecciones no asignadas
			unmatched_tracks += [r for r in range(len(self.tracks))
									if r not in row_ind]
			unmatched_dets += [c for c in range(len(det_boxes))
								if c not in col_ind]

			# 2.1) Actualizamos las pistas emparejadas
			for r, c in matched:
				self.tracks[r].update(det_boxes[c], float(det_scores[c]))

			# 2.2) Marcamos las no emparejadas
			for r in unmatched_tracks:
				self.tracks[r].mark_missed()

			# 2.3) Eliminamos las que superan max_misses
			self.tracks = [t for t in self.tracks
							if t.misses <= self.max_misses]

			# 2.4) Creamos pistas nuevas
			if len(unmatched_dets) > 0:
				self._add_tracks(det_boxes[unmatched_dets],
									det_scores[unmatched_dets])

		# 3) Guardamos resultados de este frame
		for t in self.tracks:
			self.results.setdefault(t.id, {})
			self.results[t.id][self.im_index] = np.concatenate(
				[t.box.numpy(), np.array([t.score])]
			)

		self.im_index += 1

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box.detach().cpu()
		self.score = score
		self.misses = 0         # frames consecutivos sin asociación
		self.hits = 1           # nº de veces asociada

	def update(self, box, score):
		self.box = box.detach().cpu()
		self.score = score
		self.misses = 0
		self.hits += 1

	def mark_missed(self):
		self.misses += 1

