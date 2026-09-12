# Deep Learning for Perception - Assignment 1
**FAST-NUCES (Fall 2026)**  
**Roll Number:** 23F-0620  
**Repo Link:** https://github.com/sjd-1214/DLP_ASS01.git  
**Random Seed:** 42  

Hey, this is my submission repo for Assignment 1 ("Building, Breaking and Fixing a Neural Network"). 

Everything requested in the assignment is implemented and already executed inside `DL_ASS01_23F_0620.ipynb`. All outputs and plots are visible right in the notebook, and the Word file summary is in `DL_ASS01_Summary.docx`.

---

## Quick Summary of Results

* **Final Test Score:** **88.98%** accuracy on the held-out test set (Macro Precision: 89.06%, Macro Recall: 88.98%, Macro F1: 88.94%).
* **Best Configuration:** 2 hidden layers (width 256 -> 128), ReLU, BatchNorm, Dropout 0.4, Adam with lr = 0.001, weight decay 1e-4.
* **Single Change That Helped Most:** Giving the model more training data. Scaling from 2,000 to 20,000 samples dropped the gap from 13.49% to 8.87% and pushed validation accuracy to 87.83% without hurting training accuracy. None of the penalty regularizers came close to that.

---

## How to Run / Reproduce Everything

Everything was run with `SEED = 42` so the numbers will match exactly.

### Running on Kaggle (Recommended)
1. Go to Kaggle, start a new notebook.
2. In the right panel, switch Accelerator to **GPU T4 x2**.
3. Upload `DL_ASS01_23F_0620.ipynb`.
4. Hit **Run All**. It downloads Fashion-MNIST automatically and finishes pretty fast.

### Running Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/sjd-1214/DLP_ASS01.git
   cd DLP_ASS01
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib seaborn python-docx
   ```
3. Open Jupyter:
   ```bash
   jupyter notebook DL_ASS01_23F_0620.ipynb
   ```
   Run all cells.

---

## Quick Part-by-Part Notes

* **Part 1 (NumPy Backprop):** Implemented the 2-layer MLP (784-64-10) in pure NumPy. Rebuilt it in PyTorch with identical initial weights to check gradients on the same batch. Max difference was `4.02e-08`, which proves the manual backprop is 100% correct.
* **Part 2 (Activations):** Tested Sigmoid, Tanh, ReLU, and Leaky ReLU. Sigmoid initial gradient was tiny (`0.000103` vs `0.000654` for ReLU), showing vanishing gradients. ReLU had 13 dead units (6.77%) across validation batches.
* **Part 3 (Loss Functions):** Cross-entropy vs MSE on one-hot targets. CE converges much faster because its gradient doesn't saturate when predictions are way off. Also ran California Housing regression (Test MSE: 0.2862, MAE: 0.3692).
* **Part 4 (Optimisers):** Compared SGD, Momentum, RMSProp, and Adam. Adam reached 85% val acc in 2 epochs and finished highest at 89.08% in 19 seconds. Full table is in the notebook.
* **Part 5 (Overfitting):** Cut train data to 2,000 samples and blew up the network to 4 hidden layers of 512 units. Train accuracy passed 99% (reached 99.30%), val was 83.44%, giving a 15.86% generalization gap. Curves separated around epoch 6. Classic low bias + high variance.
* **Part 6 (Regularisation):** Tested L2 (3 lambdas), L1 (88.4% weights below 1e-3), Dropout (0.2, 0.4, 0.6), BatchNorm, Early Stopping (stopped epoch 18), Data Augmentation, and more data (10k, 20k). Complete table and gap plots are in the notebook and docx.
* **Part 7 (CV & Test):** Ran 5-fold CV over 12 random configurations. Top pick was width 256, dropout 0.4, lr 0.001 (mean CV 85.03%). Retrained on full 60k data and tested once on the 10k test set: got 88.98%. That's 0.10% lower than Part 2 baseline (89.08%)—an honest negative result because heavy dropout (0.4) caused slight underfitting on confusing classes like Shirts and Pullovers when trained on all 60k samples.

---

## Files in this Repo

* `DL_ASS01_23F_0620.ipynb` - The main notebook with code, outputs, and plots for all 7 parts.
* `DL_ASS01_Summary.docx` - Word file with the one-page results summary and required tables.
* `figures/` - Saved high-res PNG plots for all parts.
* `results.json` - Saved metrics and history from the runs.
* `DLP_Assignment01.pdf` - The original assignment instructions.
