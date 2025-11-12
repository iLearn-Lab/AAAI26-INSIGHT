# Hand-Object Semantic Action Recognition

Implementation of the **Hand-Object Semantic Action Recognition** module for egocentric action anticipation.  
It fuses **frame** and **hand-object** features, aligns **verb-noun** semantics via a learned co-occurrence prior, and applies a lightweight **re-ranking** step for better rare-class recognition.

## Data Setup

Organize the extracted features and annotations as follows:

```
<frame_features_dir>/
  train/
    <clip_uid>_<action_idx>.pt
  val/
    <clip_uid>_<action_idx>.pt
  test/
    <clip_uid>_<action_idx>.pt

<mask_features_dir>/
  train/
    <clip_uid>_<action_idx>.pt
  val/
    <clip_uid>_<action_idx>.pt
  test/
    <clip_uid>_<action_idx>.pt

<annotation_dir>/
  train.json
  val.json
  fho_lta_test_unannotated.json
```

## Structure
```
.
├── main.py
├── dataset.py
├── model.py
├── train.py
└── checkpoint/
```

## How It Works
1. Frame and mask features → gated MLPs → concatenated.  
2. Transformer encoder → joint verb-noun representation.  
3. Co-occurrence re-ranking → improves rare-class accuracy.

## Quick Start
1. Edit paths in `main.py`:
   ```python
   frame_features_dir = "/path/to/frame_action"
   mask_features_dir  = "/path/to/mask_action"
   annotation_dir     = "/path/to/meta_dir"
   checkpoint_dir     = "./checkpoint"
   cooccurrence_matrix_path = "/path/to/cooccurrence_matrix.pt"
   ```
2. Run training and testing:
   ```bash
   python main.py
   ```

Outputs:
```
checkpoint/best_model.pth
predictions.csv
```

## Resume / Test
* **Resume:** keep `best_model.pth` in `checkpoint/`.
* **Test only:** comment out training and run `python main.py`.
