import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import Counter

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = None,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_verb = 0
    correct_noun = 0
    correct_action = 0

    loop = tqdm(dataloader, desc="train", leave=False)
    for frame_feats, mask_feats, verb_labels, noun_labels, _ in loop:
        frame_feats = frame_feats.to(device)
        mask_feats = mask_feats.to(device)
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)
        total_actions = frame_feats.size(0)

        optimizer.zero_grad()
        verb_logits, noun_logits = model(frame_feats, mask_feats)
        loss_verb = criterion(verb_logits, verb_labels)
        loss_noun = criterion(noun_logits, noun_labels)
        loss = loss_verb + loss_noun
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * total_actions
        total_samples += total_actions

        _, verb_preds = torch.max(verb_logits, 1)
        _, noun_preds = torch.max(noun_logits, 1)
        correct_verb += (verb_preds == verb_labels).sum().item()
        correct_noun += (noun_preds == noun_labels).sum().item()
        correct_action += ((verb_preds == verb_labels) & (noun_preds == noun_labels)).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    verb_acc = (correct_verb / total_samples) * 100
    noun_acc = (correct_noun / total_samples) * 100
    action_acc = (correct_action / total_samples) * 100

    return avg_loss, verb_acc, noun_acc, action_acc

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_verb = 0
    correct_noun = 0
    correct_action = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="val", leave=False)
        for frame_feats, mask_feats, verb_labels, noun_labels, _ in loop:
            frame_feats = frame_feats.to(device)
            mask_feats = mask_feats.to(device)
            verb_labels = verb_labels.to(device)
            noun_labels = noun_labels.to(device)
            total_actions = frame_feats.size(0)

            verb_logits, noun_logits = model(frame_feats, mask_feats)
            loss_verb = criterion(verb_logits, verb_labels)
            loss_noun = criterion(noun_logits, noun_labels)
            loss = loss_verb + loss_noun
            total_loss += loss.item() * total_actions
            total_samples += total_actions

            _, verb_preds = torch.max(verb_logits, 1)
            _, noun_preds = torch.max(noun_logits, 1)
            correct_verb += (verb_preds == verb_labels).sum().item()
            correct_noun += (noun_preds == noun_labels).sum().item()
            correct_action += ((verb_preds == verb_labels) & (noun_preds == noun_labels)).sum().item()

    avg_loss = total_loss / total_samples
    verb_acc = (correct_verb / total_samples) * 100
    noun_acc = (correct_noun / total_samples) * 100
    action_acc = (correct_action / total_samples) * 100

    return avg_loss, verb_acc, noun_acc, action_acc
