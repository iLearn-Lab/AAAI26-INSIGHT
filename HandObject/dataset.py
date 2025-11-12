import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import logging

class ActionDataset(Dataset):
    def __init__(
        self,
        splits: List[str],
        frame_features_dir: str,
        mask_features_dir: str,
        annotation_dir: str,
        transform: Optional[callable] = None,
        is_test: bool = False,
        window_size: int = 8,
        stride: int = 1
    ):
        super(ActionDataset, self).__init__()
        self.splits = [split.lower() for split in splits]
        for split in self.splits:
            assert split in ['train', 'val', 'test'],

        self.frame_features_dir = Path(frame_features_dir)
        self.mask_features_dir = Path(mask_features_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.is_test = is_test
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for split in self.splits:
            if split == 'train':
                annotation_file = self.annotation_dir / "train.json"
            elif split == 'val':
                annotation_file = self.annotation_dir / "val.json"
            elif split == 'test':
                annotation_file = self.annotation_dir / "fho_lta_test_unannotated.json"


            with open(annotation_file, 'r') as f:
                data = json.load(f)

            clip_dict = {}
            for clip in data.get('clips', []):
                clip_uid = clip['clip_uid']
                if clip_uid not in clip_dict:
                    clip_dict[clip_uid] = []
                clip_dict[clip_uid].append(clip)

            for clip_uid, clips in clip_dict.items():
                clips = sorted(clips, key=lambda x: x['action_idx'])
                if len(clips) >= self.window_size:
                    for start in range(0, len(clips) - self.window_size + 1, self.stride):
                        window_clips = clips[start:start + self.window_size]
                        window_sample = []
                        for clip in window_clips:
                            action_idx = clip['action_idx']
                            frame_path = self.frame_features_dir / split / f"{clip_uid}_{action_idx}.pt"
                            mask_path = self.mask_features_dir / split / f"{clip_uid}_{action_idx}.pt"
                            if not frame_path.exists():
                                logging.warning(f"Frame feature {frame_path} does not exist, skipping")
                                continue
                            if not self.is_test and not mask_path.exists():
                                logging.warning(f"Mask feature {mask_path} does not exist, skipping")
                                continue
                            verb_label = clip.get('verb_label', -1) if not self.is_test else -1
                            noun_label = clip.get('noun_label', -1) if not self.is_test else -1
                            window_sample.append((
                                frame_path,
                                mask_path,
                                verb_label,
                                noun_label,
                                f"{clip_uid}_{action_idx}"
                            ))
                        if len(window_sample) == self.window_size:
                            self.samples.append(window_sample)
                else:
                    for clip in clips:
                        action_idx = clip['action_idx']
                        frame_path = self.frame_features_dir / split / f"{clip_uid}_{action_idx}.pt"
                        mask_path = self.mask_features_dir / split / f"{clip_uid}_{action_idx}.pt"
                        if not frame_path.exists():
                            logging.warning(f"Frame feature {frame_path} does not exist, skipping")
                            continue
                        if not self.is_test and not mask_path.exists():
                            logging.warning(f"Mask feature {mask_path} does not exist, skipping")
                            continue
                        verb_label = clip.get('verb_label', -1) if not self.is_test else -1
                        noun_label = clip.get('noun_label', -1) if not self.is_test else -1
                        self.samples.append([(
                            frame_path,
                            mask_path,
                            verb_label,
                            noun_label,
                            f"{clip_uid}_{action_idx}"
                        )])


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        sample = self.samples[idx] 
        frame_features = []
        mask_features = []
        verb_labels = []
        noun_labels = []
        action_ids = []

        for frame_path, mask_path, verb_label, noun_label, action_id in sample:
            frame_feature = torch.load(frame_path, weights_only=True).to(dtype=torch.float32)
            mask_feature = torch.load(mask_path, weights_only=True).to(dtype=torch.float32)

            if self.transform:
                frame_feature = self.transform(frame_feature)
                mask_feature = self.transform(mask_feature)

            frame_features.append(frame_feature.unsqueeze(0))
            mask_features.append(mask_feature.unsqueeze(0))
            verb_labels.append(verb_label)
            noun_labels.append(noun_label)
            action_ids.append(action_id)

        frame_features = torch.cat(frame_features, dim=0)
        mask_features = torch.cat(mask_features, dim=0)
        verb_labels = torch.tensor(verb_labels, dtype=torch.long)
        noun_labels = torch.tensor(noun_labels, dtype=torch.long)

        return frame_features, mask_features, verb_labels, noun_labels, action_ids