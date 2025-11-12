import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
import logging
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random
import numpy as np

from dataset import ActionDataset
from model import ActionRecognitionModel
from train import train_one_epoch, validate

def setup_logging(log_file: str = 'training.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

def custom_collate_fn(batch):
    frame_feats_list = []
    mask_feats_list = []
    verb_labels_list = []
    noun_labels_list = []
    action_ids_list = []

    for sample in batch:
        frame_feats, mask_feats, verb_labels, noun_labels, action_ids = sample
        seq_len = frame_feats.size(0)
        for i in range(seq_len):
            frame_feats_list.append(frame_feats[i:i+1])
            mask_feats_list.append(mask_feats[i:i+1])
            verb_labels_list.append(verb_labels[i:i+1])
            noun_labels_list.append(noun_labels[i:i+1])
            action_ids_list.append(action_ids[i])

    frame_feats = torch.cat(frame_feats_list, dim=0)
    mask_feats = torch.cat(mask_feats_list, dim=0)
    verb_labels = torch.cat(verb_labels_list, dim=0)
    noun_labels = torch.cat(noun_labels_list, dim=0)

    return frame_feats, mask_feats, verb_labels, noun_labels, action_ids_list

def get_dataloader(
    splits: list,
    frame_features_dir: str,
    mask_features_dir: str,
    annotation_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = ActionDataset(
        splits=splits,
        frame_features_dir=frame_features_dir,
        mask_features_dir=mask_features_dir,
        annotation_dir=annotation_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

def build_cooccurrence_matrix(
    data_loader: DataLoader,
    verb_num_classes: int = 117,
    noun_num_classes: int = 521,
    alpha: float = 1.0,
    save_path: str = '.../cooccurrence_matrix.pt'
) -> dict:
    verb_counts = torch.zeros(verb_num_classes)
    noun_counts = torch.zeros(noun_num_classes)
    cooccurrence = torch.zeros(verb_num_classes, noun_num_classes)

    for frame_feats, mask_feats, verb_labels, noun_labels, _ in tqdm(data_loader, desc="build matrix"):
        mask = (verb_labels != -1) & (noun_labels != -1)
        if mask.sum() == 0:
            continue
        verb_labels = verb_labels[mask]
        noun_labels = noun_labels[mask]
        for v, n in zip(verb_labels.tolist(), noun_labels.tolist()):
            verb_counts[v] += 1
            noun_counts[n] += 1
            cooccurrence[v, n] += 1

    P_n_given_v = (cooccurrence + alpha) / (verb_counts[:, None] + alpha * noun_num_classes)
    P_v_given_n = (cooccurrence.t() + alpha) / (noun_counts[:, None] + alpha * verb_num_classes)
    cooccurrence_sum = cooccurrence.sum()
    O = cooccurrence / cooccurrence_sum if cooccurrence_sum > 0 else torch.ones_like(cooccurrence) / cooccurrence.numel()

    matrix_dict = {
        'P_n_given_v': P_n_given_v,
        'P_v_given_n': P_v_given_n,
        'O': O,
        'raw_cooccurrence': cooccurrence
    }
    torch.save(matrix_dict, save_path)
    logging.info(f"matrix saved in {save_path}")
    return matrix_dict

def load_cooccurrence_matrix(
    load_path: str = '.../cooccurrence_matrix.pt'
) -> dict:
    if os.path.exists(load_path):
        matrix_dict = torch.load(load_path)
        logging.info(f"matrix loaded from {load_path}")
        return matrix_dict
    else:
        logging.error(f"matrix is none in {load_path}")
        raise FileNotFoundError(f"matrix is none in {load_path}")

def find_best_verb_noun_pair(
    top5_verbs, top5_verb_probs, top5_nouns, top5_noun_probs, cooccurrence_matrix
):
    co_mat = cooccurrence_matrix['raw_cooccurrence']
    verb_probs_tensor = torch.tensor(top5_verb_probs, dtype=torch.float32)
    noun_probs_tensor = torch.tensor(top5_noun_probs, dtype=torch.float32)
    verb_probs_normalized = verb_probs_tensor / verb_probs_tensor.sum()
    noun_probs_normalized = noun_probs_tensor / noun_probs_tensor.sum()

    max_prob = -1
    best_v = top5_verbs[0]
    best_n = top5_nouns[0]
    for i, v in enumerate(top5_verbs):
        for j, n in enumerate(top5_nouns):
            if co_mat[v, n] > 0:
                joint_prob = verb_probs_normalized[i] * noun_probs_normalized[j]
                if joint_prob > max_prob:
                    max_prob = joint_prob
                    best_v = v
                    best_n = n

    if max_prob > 0:
        return best_v, best_n
    else:
        return top5_verbs[0], top5_nouns[0]

def validate_with_cooccurrence(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cooccurrence_matrix: dict,
):
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
            mask = (verb_labels != -1) & (noun_labels != -1)
            if mask.sum() > 0:
                loss_verb = criterion(verb_logits[mask], verb_labels[mask])
                loss_noun = criterion(noun_logits[mask], noun_labels[mask])
                loss = loss_verb + loss_noun
                total_loss += loss.item() * mask.sum().item()
                total_samples += mask.sum().item()

            verb_probs = torch.softmax(verb_logits, dim=1)
            noun_probs = torch.softmax(noun_logits, dim=1)
            _, top5_verb_indices = torch.topk(verb_probs, 5, dim=1)
            _, top5_noun_indices = torch.topk(noun_probs, 5, dim=1)
            top5_verb_probs = torch.gather(verb_probs, 1, top5_verb_indices)
            top5_noun_probs = torch.gather(noun_probs, 1, top5_noun_indices)

            for j in range(total_actions):
                if not mask[j]:
                    continue
                tv = top5_verb_indices[j].tolist()
                tpv = top5_verb_probs[j].tolist()
                tn = top5_noun_indices[j].tolist()
                tpn = top5_noun_probs[j].tolist()

                best_v, best_n = find_best_verb_noun_pair(tv, tpv, tn, tpn, cooccurrence_matrix)
                true_v = verb_labels[j].item()
                true_n = noun_labels[j].item()
                correct_verb += (best_v == true_v)
                correct_noun += (best_n == true_n)
                correct_action += (best_v == true_v and best_n == true_n)

            loop.set_postfix(loss=total_loss / total_samples if total_samples > 0 else 0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    verb_acc = (correct_verb / total_samples) * 100 if total_samples > 0 else 0.0
    noun_acc = (correct_noun / total_samples) * 100 if total_samples > 0 else 0.0
    action_acc = (correct_action / total_samples) * 100 if total_samples > 0 else 0.0

    return avg_loss, verb_acc, noun_acc, action_acc

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler_plateau: ReduceLROnPlateau,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str,
    cooccurrence_matrix: dict,
    patience: int = 5,
    max_grad_norm: float = 1.0,
):
    model.to(device)
    model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_action_accuracy = -1.0
    best_epoch = -1
    best_model_path = Path(checkpoint_dir) / "best_model.pth"
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_verb_acc, train_noun_acc, train_action_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm
        )
        val_loss, val_verb_acc, val_noun_acc, val_action_acc = validate_with_cooccurrence(
            model, val_loader, criterion, device, cooccurrence_matrix
        )
        scheduler_plateau.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")

        logging.info(
            f"Epoch [{epoch}/{epochs}] finish:\n"
            f"  Train Loss: {train_loss:.4f}, Verb Acc: {train_verb_acc:.2f}%, Noun Acc: {train_noun_acc:.2f}%, Action Acc: {train_action_acc:.2f}%\n"
            f"  Val Loss: {val_loss:.4f}, Verb Acc: {val_verb_acc:.2f}%, Noun Acc: {val_noun_acc:.2f}%, Action Acc: {val_action_acc:.2f}%"
        )

        if val_action_acc > best_action_accuracy:
            best_action_accuracy = val_action_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            break

        if epoch % 10 == 0:
            torch.save(model.state_dict(), Path(checkpoint_dir) / f"model_epoch_{epoch}.pth")



def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    cooccurrence_matrix: dict,
    output_csv: str = 'test_predictions.csv',
):
    model.eval()
    action_predictions = {}

    with torch.no_grad():
        for frame_feats, mask_feats, _, _, action_ids in tqdm(test_loader, desc="test"):
            frame_feats = frame_feats.to(device)
            mask_feats = mask_feats.to(device)
            total_actions = frame_feats.size(0)

            verb_logits, noun_logits = model(frame_feats, mask_feats)
            verb_probs = torch.softmax(verb_logits, dim=-1)
            noun_probs = torch.softmax(noun_logits, dim=-1)

            _, top5_verb_indices = torch.topk(verb_probs, 5, dim=-1)
            _, top5_noun_indices = torch.topk(noun_probs, 5, dim=-1)
            top5_verb_probs = torch.gather(verb_probs, -1, top5_verb_indices)
            top5_noun_probs = torch.gather(noun_probs, -1, top5_noun_indices)

            for j in range(total_actions):
                action_id = action_ids[j]
                tv = top5_verb_indices[j].tolist()
                tpv = top5_verb_probs[j].tolist()
                tn = top5_noun_indices[j].tolist()
                tpn = top5_noun_probs[j].tolist()

                best_v, best_n = find_best_verb_noun_pair(tv, tpv, tn, tpn, cooccurrence_matrix)
                entry = action_predictions.setdefault(action_id, {
                    'verbs': [], 'nouns': [], 'verb_probs': [], 'noun_probs': []
                })
                entry['verbs'].append(best_v)
                entry['nouns'].append(best_n)
                entry['verb_probs'].append(max(tpv))
                entry['noun_probs'].append(max(tpn))

    final_predictions = []
    for action_id, entry in action_predictions.items():
        verb_counts = Counter(entry['verbs'])
        noun_counts = Counter(entry['nouns'])
        most_common_verbs = verb_counts.most_common()
        most_common_nouns = noun_counts.most_common()

        if len(most_common_verbs) > 1 and most_common_verbs[0][1] == most_common_verbs[1][1]:
            max_prob, best_verb = -1, most_common_verbs[0][0]
            for verb, count in most_common_verbs:
                if count == most_common_verbs[0][1]:
                    avg_p = sum(p for v, p in zip(entry['verbs'], entry['verb_probs']) if v == verb) / count
                    if avg_p > max_prob:
                        max_prob, best_verb = avg_p, verb
            most_common_verb = best_verb
        else:
            most_common_verb = most_common_verbs[0][0]

        if len(most_common_nouns) > 1 and most_common_nouns[0][1] == most_common_nouns[1][1]:
            max_prob, best_noun = -1, most_common_nouns[0][0]
            for noun, count in most_common_nouns:
                if count == most_common_nouns[0][1]:
                    avg_p = sum(p for n, p in zip(entry['nouns'], entry['noun_probs']) if n == noun) / count
                    if avg_p > max_prob:
                        max_prob, best_noun = avg_p, noun
            most_common_noun = best_noun
        else:
            most_common_noun = most_common_nouns[0][0]

        final_predictions.append({
            'clip_uid_action_id': action_id,
            'verb_label': most_common_verb,
            'noun_label': most_common_noun
        })

    pd.DataFrame(final_predictions).to_csv(output_csv, index=False)

def main():
    setup_logging()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    frame_features_dir = ".../frame_action"
    mask_features_dir = ".../mask_action"
    annotation_dir = ".../meta_dir"
    checkpoint_dir = "./checkpoint"
    cooccurrence_matrix_path = ".../cooccurrence_matrix.pt"

    batch_size = 8
    epochs = 40
    learning_rate = 8e-5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 1408
    mlp_hidden_dim = 2048
    mlp_output_dim = 256
    transformer_layers = 4
    n_heads = 8
    transformer_hidden_dim = 2048
    verb_num_classes = 117
    noun_num_classes = 521



    if os.path.exists(cooccurrence_matrix_path):
        cooccurrence_matrix = load_cooccurrence_matrix(cooccurrence_matrix_path)
    else:
        combined_loader = get_dataloader(
            splits=['train'],
            frame_features_dir=frame_features_dir,
            mask_features_dir=mask_features_dir,
            annotation_dir=annotation_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        cooccurrence_matrix = build_cooccurrence_matrix(combined_loader, save_path=cooccurrence_matrix_path)

    train_loader = get_dataloader(
        splits=['train'],
        frame_features_dir=frame_features_dir,
        mask_features_dir=mask_features_dir,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        splits=['val'],
        frame_features_dir=frame_features_dir,
        mask_features_dir=mask_features_dir,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = get_dataloader(
        splits=['test'],
        frame_features_dir=frame_features_dir,
        mask_features_dir=mask_features_dir,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = ActionRecognitionModel(
        input_dim=input_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_output_dim=mlp_output_dim,
        transformer_layers=transformer_layers,
        n_heads=n_heads,
        transformer_hidden_dim=transformer_hidden_dim,
        verb_num_classes=verb_num_classes,
        noun_num_classes=noun_num_classes,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pth")
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pth")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        if os.path.exists(scheduler_path):
            plateau_scheduler.load_state_dict(torch.load(scheduler_path))
        start_epoch = 1

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_plateau=plateau_scheduler,
        device=device,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        cooccurrence_matrix=cooccurrence_matrix,
        patience=5,
        max_grad_norm=1.0,
    )

    best_model_path = Path(checkpoint_dir) / "best_model.pth"
    if not best_model_path.exists():
        return
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    test(
        model=model,
        test_loader=test_loader,
        device=device,
        cooccurrence_matrix=cooccurrence_matrix,
        output_csv="test_predictions.csv",
    )

if __name__ == "__main__":
    main()
