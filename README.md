# Transfer Learning: Oxford Flowers-102

**Dataset:** Oxford Flowers-102 (102 fine-grained flower classes, `torchvision.datasets.Flowers102`, auto-download)
**Models:** All seven architectures from the assignment's Model Options table are trained and compared — ResNet18, ResNet50, VGG16, DenseNet121, EfficientNetV2-S, MobileNetV3-Large, ConvNeXt-Tiny — with the best one selected based on validation accuracy.

Each architecture is run with both transfer learning strategies:
- **Feature extraction** — backbone frozen, only a new classification head is trained
- **Fine-tuning** — backbone unfrozen and trained at a low learning rate alongside the head at a higher learning rate

## Project Structure

```
transfer-learning-flower-classification/
├── data/                     # Flowers-102 auto-downloads here
├── notebooks/
│   └── transfer_learning.ipynb   # Main deliverable — orchestrates the full sweep
├── saved_models/             # Trained weights, one per (model, strategy) + best overall
├── results/
│   ├── plots/                # Training curves, leaderboard chart
│   ├── confusion_matrix/     # Confusion matrix for the winning model
│   └── predictions/          # Raw test-set predictions (CSV)
├── src/
│   ├── dataset.py            # Task 1 — transforms & DataLoaders
│   ├── model.py               # Task 2 — model construction / head swap
│   ├── train.py               # Task 3–4 — training loop & full model sweep
│   ├── evaluate.py            # Task 3/5 — evaluation, predictions, plots
│   └── utils.py                # Seeding, device, checkpoints, model registry, class names
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab: **Runtime → Change runtime type → GPU**, then `pip install -r requirements.txt` in the first cell if needed (torch/torchvision are usually preinstalled).

## Running

Open `notebooks/transfer_learning.ipynb` and run all cells top to bottom. It will:

1. Sanity-check the data pipeline on one model
2. Call `run_two_phase_sweep()` from `src/train.py`:
   - **Phase 1 (screen):** all 7 models, feature extraction only, a few epochs each — cheap, just to rank candidates
   - **Phase 2 (full):** only the top `TOP_K` models from the screen get fully trained with both feature extraction and fine-tuning
3. Rank every (model, strategy) combo by validation accuracy
4. Automatically select the best-performing combo
5. Run the full Task 5 analysis (curves, classification report, confusion matrix, most-confused classes) on the winner
6. Save the winning model's final weights

This two-phase approach avoids training all 7 models × 2 strategies (14 full runs) end-to-end, which is slow — instead it's 7 cheap screens + a handful of full runs on the shortlist. If you have the time/compute and want the exhaustive version instead, use `run_all_experiments()` directly (a commented-out example is in the sweep cell).

Adjust `EPOCHS`, `SCREEN_EPOCHS`, `TOP_K`, `BATCH_SIZE`, `HEAD_LR`, `BACKBONE_LR` at the top of the sweep cell. Start with small values (`SCREEN_EPOCHS=2-3`, `EPOCHS=5`) to confirm everything runs end-to-end before committing to a longer run, since VGG16/ResNet50/ConvNeXt-Tiny fine-tuning will still be noticeably slower than MobileNetV3/ResNet18 even in phase 2.

## Results

See `results/leaderboard.json` after running for the full ranked comparison, and the reflection-question cells at the end of the notebook for the writeup.
