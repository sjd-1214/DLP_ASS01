# Deep Learning for Perception - Assignment 1
**FAST-NUCES (Fall 2026)**  
**Roll Number:** 23F-0620  
**Repo Link:** https://github.com/sjd-1214/DLP_ASS01.git  
**Random Seed:** 42  

Hey, this is my submission repo for Assignment 1 ("Building, Breaking and Fixing a Neural Network"). 

The dataset is loaded directly from the `Dataset/` folder (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`). Everything requested in the assignment is implemented and already executed inside `DL_ASS01_23F_0620.ipynb`. All outputs and plots are visible right in the notebook, and the Word file summary is in `DL_ASS01_Summary.docx`.

---

## Quick Summary of Results

* **Final Test Score:** **90.45%** accuracy on the held-out test set (Macro Precision: 90.67%, Macro Recall: 90.45%, Macro F1: 90.52%).
* **Best Configuration:** 2 hidden layers (width 512 -> 256), ReLU, BatchNorm, Dropout 0.2, Adam with lr = 0.0005, weight decay 1e-4.
* **Single Change That Helped Most:** Giving the model more training data. Scaling from 2,000 to 20,000 samples dropped the gap from 13.91% to 8.67% and pushed validation accuracy to 87.58% without hurting training accuracy. None of the penalty regularizers came close to that.
* **Improvement over Baseline:** +2.15 percentage points higher than the Part 2 baseline (88.30% -> 90.45%).

---

## How to Run / Reproduce Everything

Everything was run with `SEED = 42` so the numbers will match exactly.

### Running on Kaggle (Recommended)
1. Go to Kaggle, start a new notebook.
2. In the right panel, switch Accelerator to **GPU T4 x2**.
3. Upload `DL_ASS01_23F_0620.ipynb` and ensure the `Dataset/` folder (or Kaggle fashionmnist dataset) is attached.
4. Hit **Run All**. The notebook auto-detects the dataset path and finishes pretty fast.

### Running Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/sjd-1214/DLP_ASS01.git
   cd DLP_ASS01
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn python-docx
   ```
3. Open Jupyter:
   ```bash
   jupyter notebook DL_ASS01_23F_0620.ipynb
   ```
   Run all cells.

---

## Quick Part-by-Part Notes

* **Part 1 (NumPy Backprop):** Implemented the 2-layer MLP (784-64-10) in pure NumPy. Rebuilt it in PyTorch with identical initial weights to check gradients on the same batch. Max difference was `2.88e-08`, which proves the manual backprop is 100% correct.
* **Part 2 (Activations):** Tested Sigmoid, Tanh, ReLU, and Leaky ReLU. Sigmoid initial gradient was tiny (`0.000095` vs `0.000662` for ReLU), showing vanishing gradients. ReLU had 19 dead units (9.90%) across validation batches. Baseline validation accuracy was 88.30%.
* **Part 3 (Loss Functions):** Cross-entropy vs MSE on one-hot targets. CE converges faster because its gradient doesn't saturate when predictions are way off. Also ran California Housing regression (Test MSE: 0.2862, MAE: 0.3692).
* **Part 4 (Optimisers):** Compared SGD, Momentum, RMSProp, and Adam. Adam reached 85% val acc in 2 epochs. Full table is in the notebook.
* **Part 5 (Overfitting):** Cut train data to 2,000 samples and blew up the network to 4 hidden layers of 512 units. Train accuracy passed 99% (reached 99.45%), val was 82.74%, giving a 16.71% generalization gap. Curves separated around epoch 7. Classic low bias + high variance.
* **Part 6 (Regularisation):** Tested L2 (3 lambdas), L1 (89.7% weights below 1e-3), Dropout (0.2, 0.4, 0.6), BatchNorm, Early Stopping (stopped epoch 12), Data Augmentation, and more data (10k, 20k). Complete table and gap plots are in the notebook and docx.
* **Part 7 (CV & Test):** Ran 5-fold CV over 12 random configurations. Top pick was width 512, dropout 0.2, lr 0.0005 (mean CV 85.11%). Retrained on full 60k data and tested once on the 10k test set: got **90.45%**, an improvement of +2.15 percentage points over the Part 2 baseline (88.30%).

---

## Files in this Repo

* `DL_ASS01_23F_0620.ipynb` - The main notebook with code, outputs, and plots for all 7 parts.
* `DL_ASS01_Summary.docx` - Word file with the one-page results summary and required tables.
* `Dataset/` - Fashion-MNIST CSVs (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) and ubyte files.
* `figures/` - Saved high-res PNG plots for all parts.
* `DLP_Assignment01.pdf` - The original assignment instructions.
