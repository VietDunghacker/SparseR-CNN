# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import operator
import torch
import torch.utils.data
import json
from detectron2.utils.comm import get_world_size

from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
from collections import defaultdict
from typing import Optional

# from .custom_build_augmentation import build_custom_augmentation

def build_custom_train_loader(cfg, mapper=None):
	"""
	Modified from detectron2.data.build.build_custom_train_loader, but supports
	different samplers
	"""
	if True:
		dataset_dicts = get_detection_dataset_dicts_with_source(
			cfg.DATASETS.TRAIN,
			filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
			min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
			if cfg.MODEL.KEYPOINT_ON
			else 0,
			proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
		)
		sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
		for d in dataset_dicts:
			sizes[d['dataset_source']] += 1
		print('dataset sizes', sizes)
	else:
		dataset_dicts = get_detection_dataset_dicts(
			cfg.DATASETS.TRAIN,
			filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
			min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
			if cfg.MODEL.KEYPOINT_ON
			else 0,
			proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
		)
	dataset = DatasetFromList(dataset_dicts, copy=False)

	if mapper is None:
		assert 0
		# mapper = DatasetMapper(cfg, True)
	dataset = MapDataset(dataset, mapper)

	sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
	logger = logging.getLogger('detectron2')
	logger.info("Using training sampler {}".format(sampler_name))
	# TODO avoid if-else?
	if sampler_name == "TrainingSampler":
		sampler = TrainingSampler(len(dataset))
	elif sampler_name == "MultiDatasetSampler":
		assert source_aware
		sampler = MultiDatasetSampler(cfg, sizes, dataset_dicts)
	elif sampler_name == "RepeatFactorTrainingSampler":
		repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
			dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
		)
		sampler = RepeatFactorTrainingSampler(repeat_factors)
	elif sampler_name == "ClassAwareSampler":
		sampler = ClassAwareSampler(dataset_dicts, cfg)
	else:
		raise ValueError("Unknown training sampler: {}".format(sampler_name))

	return build_batch_data_loader(
		dataset,
		sampler,
		cfg.SOLVER.IMS_PER_BATCH,
		aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
		num_workers=cfg.DATALOADER.NUM_WORKERS,
	)


class ClassAwareSampler(Sampler):
	def __init__(self, dataset_dicts, cfg):
		"""
		Args:
			size (int): the total number of data of the underlying dataset to sample from
			seed (int): the initial seed of the shuffle. Must be the same
				across all workers. If None, will use a random seed shared
				among workers (require synchronization among all workers).
		"""
		self.dataset_dicts = dataset_dicts
		self._size = cfg.SOLVER.IMS_PER_BATCH
		assert self._size > 0
		
		self._rank = comm.get_rank()
		self._world_size = comm.get_world_size()

		category_freq = defaultdict(int)
		self.cw = []
		for dataset_dict in dataset_dicts:  # For each image (without repeats)
			cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
			for cat_id in cat_ids:
				category_freq[cat_id] += 1
		for i in sorted(category_freq.keys()):
			self.cw.append(1. / category_freq[i])
		self.cw = np.array(self.cw)
		self.cw /= sum(self.cw)

		self.weights = self._get_class_balance_factor(dataset_dicts)

	def __iter__(self):
		start = self._rank
		yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)


	def _infinite_indices(self):
		while True:
			ids = torch.multinomial(self.weights, self._size, replacement=True)
			yield from ids
			self.weights = self._get_class_balance_factor(self.dataset_dicts)

	def _get_class_balance_factor(self, dataset_dicts):
		# 1. For each category c, compute the fraction of images that contain it: f(c)
		ret = []
		for i, dataset_dict in enumerate(dataset_dicts):
			cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
			ret.append(sum([self.cw[cat_id] for cat_id in cat_ids]))
		return torch.tensor(ret).float()


def get_detection_dataset_dicts_with_source(
	dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
	assert len(dataset_names)
	dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
	for dataset_name, dicts in zip(dataset_names, dataset_dicts):
		assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
	
	for source_id, (dataset_name, dicts) in \
		enumerate(zip(dataset_names, dataset_dicts)):
		assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
		for d in dicts:
			d['dataset_source'] = source_id

		if "annotations" in dicts[0]:
			try:
				class_names = MetadataCatalog.get(dataset_name).thing_classes
				check_metadata_consistency("thing_classes", dataset_name)
				print_instances_class_histogram(dicts, class_names)
			except AttributeError:  # class names are not available for this dataset
				pass

	assert proposal_files is None

	dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

	has_instances = "annotations" in dataset_dicts[0]
	if filter_empty and has_instances:
		dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
	if min_keypoints > 0 and has_instances:
		dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

	return dataset_dicts

class MultiDatasetSampler(Sampler):
	def __init__(self, cfg, sizes, dataset_dicts, seed: Optional[int] = None):
		"""
		Args:
			size (int): the total number of data of the underlying dataset to sample from
			seed (int): the initial seed of the shuffle. Must be the same
				across all workers. If None, will use a random seed shared
				among workers (require synchronization among all workers).
		"""
		self.sizes = sizes
		dataset_ratio = cfg.DATALOADER.DATASET_RATIO
		self._batch_size = cfg.SOLVER.IMS_PER_BATCH
		assert len(dataset_ratio) == len(sizes), \
			'length of dataset ratio {} should be equal to number if dataset {}'.format(
				len(dataset_ratio), len(sizes)
			)
		if seed is None:
			seed = comm.shared_random_seed()
		self._seed = int(seed)
		self._rank = comm.get_rank()
		self._world_size = comm.get_world_size()
		
		self._ims_per_gpu = self._batch_size // self._world_size
		self.dataset_ids =  torch.tensor(
			[d['dataset_source'] for d in dataset_dicts], dtype=torch.long)

		dataset_weight = [torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio) \
			for i, (r, s) in enumerate(zip(dataset_ratio, sizes))]
		dataset_weight = torch.cat(dataset_weight)
		self.weights = dataset_weight
		self.sample_epoch_size = len(self.weights)

	def __iter__(self):
		start = self._rank
		yield from itertools.islice(
			self._infinite_indices(), start, None, self._world_size)


	def _infinite_indices(self):
		g = torch.Generator()
		g.manual_seed(self._seed)
		while True:
			ids = torch.multinomial(
				self.weights, self.sample_epoch_size, generator=g, 
				replacement=True)
			nums = [(self.dataset_ids[ids] == i).sum().int().item() \
				for i in range(len(self.sizes))]
			print('_rank, len, nums', self._rank, len(ids), nums, flush=True)
			# print('_rank, len, nums, self.dataset_ids[ids[:10]], ', 
			#	 self._rank, len(ids), nums, self.dataset_ids[ids[:10]], 
			#	 flush=True)
			yield from ids