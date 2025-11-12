# Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation

> **TL;DR:** This framework connects perceptual understanding and cognitive reasoning to achieve egocentric video forecasting:
Perception (Hand–Object) → Reasoning (Cognitive Anticipation) → Future Action Prediction

## 🧩 Project Structure
```
.
├── HandObject/                 
├── CognitiveReasoning/          
├── pretrain_model/            
└── data/                      
```

The HandObject and CognitiveReasoning modules have their own detailed README files and scripts for training and evaluation.

## 🛠️ Installation

### 1. Create Environment
```bash
conda create -n your_env python=3.10 pip -y
conda activate your_env
```

### 2. Install PyTorch (CUDA 12.4)
```bash
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

> For CPU-only users:  
> `conda install -y pytorch torchvision torchaudio cpuonly -c pytorch`

### 3. Install Dependencies
```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift && pip install -e .
pip install deepspeed flash-attn --no-build-isolation
pip install -r requirements.txt
```

If you encounter:
```
ImportError: libGL.so.1: cannot open shared object file
```
fix it with:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6
```

### 4. Verify
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
## 🎯 Pretrained Models
Download and store all pretrained weights under:
```
/pretrain_model/
  ├── EgoVideo/
  ├── all-MiniLM-L6-v2/
  ├── SAM2/
  ├── Hand Object Detector/
  └── Qwen2.5-VL/
```

## 📂 Data Preparation

### 1. Datasets
Supports **Ego4D**, **EPIC-Kitchens-55**, and **EGTEA Gaze+**.  
Follow the official download instructions and organize under `./data`.

### 2. Feature Extraction

Run **Hand Object Detector**, **SAM2**, and **EgoVideo** in sequence, following their **official tutorials**, to obtain both **frame features** and **HOI (hand-object) features**.

## 🚀 Running the Pipeline

1. **Stage 1 - Hand-Object Semantic Action Recognition**  
   Extracts and fuses HOI and frame features.  
   → See `HandObject/README.md` for training and testing.

2. **Stage 2 - Explicit Cognitive Reasoning for Anticipation**  
   Fine-tunes Qwen2.5-VL-7B with GRPO reinforcement learning.  
   → See `CognitiveReasoning/README.md` for details.


## 🙏 Acknowledgements

We thank the authors of **[Ego4D](https://ego4d-data.org/)**, **[EPIC-Kitchens-55](https://epic-kitchens.github.io/2025)**, and **[EGTEA Gaze+](https://cbs.ic.gatech.edu/fpv/)** for providing the open-source datasets that support our experiments.  

We also thank the developers of **[Hand Object Detector](https://github.com/ddshan/hand_object_detector)**, **[SAM2](https://github.com/facebookresearch/sam2)**, and **[EgoVideo](https://github.com/OpenGVLab/EgoVideo)** for their released pretrained models and codebases.  

Finally, we acknowledge the **[ms-swift](https://github.com/modelscope/ms-swift)** framework for enabling efficient GRPO-based reinforcement learning in our cognitive reasoning module.  

We also invite readers to check out our **challenge report**, which achieved **1st place** in the *Long-Term Action Anticipation, Ego4D Challenge @ CVPR 2025*:  
🔗 [Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation (arXiv:2506.02550)](https://arxiv.org/abs/2506.02550)


## 🔖 License
[MIT License]()