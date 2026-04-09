# Cervical Cancer Detection — Advanced DL Pipeline
### SIPaKMeD | EfficientNet-B4 + ResNet50 + DenseNet201 + Ensemble | Streamlit Dashboard

---

## QUICK START

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place SIPaKMeD dataset folder next to the scripts (or configure Kaggle)
#    Expected: ./SIPaKMeD/im_Dyskeratotic/, ./SIPaKMeD/im_Koilocytotic/, etc.

# 3. Train all models (takes 1-3 hours on GPU)
python cervical_cancer_train.py

# 4. Resume if interrupted
python cervical_cancer_train.py --resume

# 5. Launch the dashboard
streamlit run cervical_cancer_app.py
```

---

## WHAT GETS TRAINED

| Model           | Architecture          | Target Accuracy |
|-----------------|-----------------------|-----------------|
| EfficientNet-B4 | timm pretrained + FT  | 97–99%          |
| ResNet50        | IMAGENET V2 + head    | 95–97%          |
| DenseNet201     | IMAGENET V1 + head    | 95–97%          |
| **Ensemble**    | Soft-vote (acc-based) | **98–99%+**     |

---

## FEATURES

### Training
- Label smoothing loss with class weights
- MixUp + CutMix augmentation (random per batch)
- Cosine annealing with warm restarts
- Mixed precision (AMP) on GPU
- Weighted sampler for imbalanced classes
- Test-Time Augmentation (TTA) for final evaluation
- Early stopping with configurable patience

### Checkpoints
- `best.pth` — best validation accuracy
- `latest.pth` — most recent epoch (for resuming)
- `epoch_XXX.pth` — every 5 epochs
- `checkpoint_log.json` — full history per model
- Resume training with `--resume` flag

### Outputs (saved to ./cervical_output/)
```
cervical_output/
├── checkpoints/
│   ├── EfficientNet_B4/
│   │   ├── best.pth
│   │   ├── latest.pth
│   │   ├── epoch_005.pth ...
│   │   └── checkpoint_log.json
│   ├── ResNet50/  ...
│   └── DenseNet201/  ...
├── results/
│   ├── EfficientNet_B4_results.json
│   ├── ResNet50_results.json
│   ├── DenseNet201_results.json
│   └── Ensemble_results.json
├── plots/
│   ├── *_history.png
│   ├── *_confusion_matrix.png
│   ├── *_roc.png
│   └── model_comparison.png
├── gradcam/
│   ├── EfficientNet_B4_gradcam_gallery.png
│   ├── EfficientNet_B4/  (individual per-image analyses)
│   └── ...
└── summary.json
```

### Streamlit Dashboard Pages
| Page              | Content                                              |
|-------------------|------------------------------------------------------|
| Overview          | Leaderboard, class stats, dataset summary            |
| Model Results     | Metrics, confusion matrix, ROC, per-class report     |
| Predict Image     | Upload cell image → prediction + Grad-CAM            |
| Grad-CAM Gallery  | Browse all saved heatmaps from training              |
| Checkpoints       | View all saved checkpoints, epoch history            |
| Training Curves   | Overlaid accuracy/loss plots for all models          |

---

## HARDWARE TIPS

**GPU (recommended)**  → Keep default settings. ~1-2hr training.

**CPU only** → Edit Config in `cervical_cancer_train.py`:
```python
NUM_EPOCHS  = 20      # Reduce from 60
BATCH_SIZE  = 16      # Reduce from 32
MODELS_TO_TRAIN = ["EfficientNet_B4"]  # Train one model only
TTA_STEPS   = 2       # Reduce from 5
```

**Low VRAM** → Reduce `BATCH_SIZE` to 8 or 16.

---

## TROUBLESHOOTING

**"No images found"**  
→ Make sure SIPaKMeD subfolders are named: `im_Dyskeratotic`, `im_Koilocytotic`, `im_Metaplastic`, `im_Parabasal`, `im_Superficial-Intermediate`

**CUDA out of memory**  
→ Set `BATCH_SIZE = 8` in Config

**timm not found**  
→ `pip install timm` — script falls back to torchvision automatically

**Windows multiprocessing error**  
→ `NUM_WORKERS` is already 0 (safe default for Windows/PyCharm)

**Streamlit won't load models**  
→ Make sure `./cervical_output/checkpoints/<ModelName>/best.pth` exists
