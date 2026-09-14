# Deep Learning for Perception - Assignment 1
**FAST-NUCES (Fall 2026)**  
**Roll Number:** 23F-0620  
**Repo Link:** https://github.com/sjd-1214/DLP_ASS01.git  
**Random Seed:** 42  

## Assignment Overview: Building, Breaking and Fixing a Neural Network

The goal of this assignment is to understand the complete practical behavior of feedforward neural networks by going through three main stages:
1. **Building:** Constructing an MLP from scratch in pure NumPy without autograd, deriving the backprop equations manually, and verifying the gradients against PyTorch. Then rebuilding it in PyTorch to study activations, loss functions, and optimizers.
2. **Breaking:** Intentionally pushing the network into extreme overfitting by starving it of data (cutting down to 2,000 samples) and massively expanding its capacity (4 hidden layers of 512 units) until training accuracy exceeds 99% with a large generalization gap.
3. **Fixing:** Applying regularization and tuning methods covered in class one by one (L2 weight decay, L1 penalty, Dropout, Batch Normalization, Early Stopping, Data Augmentation, and scaling training data) to see what actually works. Finally, tuning the model using 5-fold cross-validation on a hyperparameter grid and evaluating strictly once on the held-out test set.

Everything is implemented and executed in `DL_ASS01_23F_0620.ipynb` using the provided Fashion-MNIST dataset from the `Dataset/` folder.

---

## Part-by-Part Notes & Findings

* **Environment Setup:** Loaded `fashion-mnist_train.csv` (60,000 samples) and `fashion-mnist_test.csv` (10,000 samples) from `Dataset/`. Normalized pixel values to 0-1 and flattened images to 784-dimensional vectors. Kept an 80/20 train/validation split (48,000 train, 12,000 validation) and kept the 10,000 test set completely untouched until Part 7.

* **Part 1 (Backpropagation From Scratch):** Built a two-layer MLP (784 -> 64 -> 10) in pure NumPy with ReLU in the hidden layer and Softmax at the output. Derived manual gradients for categorical cross-entropy. Trained on a 5,000 sample subset for 20 epochs down to loss 0.3969. Copied initial weights into an identical PyTorch model to check gradients on the same batch—the max absolute difference was `2.88e-08`, confirming the math and NumPy backprop code are 100% correct.

* **Part 2 (Baseline Model & Activation Study):** Built a 2-hidden-layer PyTorch network (784 -> 128 -> 64 -> 10) and trained 4 versions comparing Sigmoid, Tanh, ReLU, and Leaky ReLU. Sigmoid initial gradient was tiny (`0.000095` vs `0.000662` for ReLU), showing clear vanishing gradients due to Sigmoid's max derivative of 0.25. ReLU had 19 dead units (9.90%) across the validation set, showing that some units get trapped in the negative zone with zero gradient, though the majority stay active. Baseline validation accuracy was 88.30%.

* **Part 3 (Loss Functions):** Compared Categorical Cross-Entropy (89.14% val acc) against Mean Squared Error on one-hot targets (89.00% val acc). Cross-entropy converges faster and cleaner because its gradient is simply (predicted - actual), maintaining strong updates even on large errors. With MSE, the gradient gets multiplied by the softmax derivative, which approaches zero when the model is confident but wrong, causing training to stall. Also trained a small MLP regressor on California Housing, achieving test MSE = 0.2862, RMSE = 0.5350, MAE = 0.3692.

* **Part 4 (Optimiser Comparison):** Compared plain SGD, SGD with momentum (0.9), RMSProp, and Adam on the baseline architecture. Adam was the fastest to reach 85% validation accuracy (only 2 epochs) and converged smoothly throughout. Its adaptive learning rates and momentum make it the best choice without needing delicate learning rate schedules.

* **Part 5 (Forcing Overfitting):** Deliberately broke the model by training on only 2,000 samples with 4 hidden layers of 512 units (over 1.2M parameters). Trained for 75 epochs until training accuracy reached 99.45% while validation accuracy stalled at 82.74% (generalization gap = 16.71%). The loss curves separated at Epoch 7, after which validation loss began climbing while training loss kept dropping. Diagnosed as low bias (high capacity fits training data easily) and severe high variance (memorizing noise).

* **Part 6 (Regularisation Study):** Tested regularizers one at a time on the overfitted setup:
  * L2 weight decay at 3 lambdas (1e-4, 1e-3, 1e-2): reduced the gap from 13.91% to 10.05%, but mainly by reducing training accuracy.
  * L1 penalty (1e-4): created sparsity with 89.7% of weights below 1e-3.
  * Dropout (rates 0.2, 0.4, 0.6): p=0.4 gave the best algorithmic balance (gap 10.38%, train acc 93.35%).
  * Batch Normalization: improved training stability but alone didn't stop overfitting on 2k samples.
  * Early stopping (patience 5): halted training at epoch 12 to prevent further divergence.
  * Data Augmentation (flips + rotations): reduced variance and helped generalization.
  * More Training Data (10k and 20k): by far the most effective fix. Expanding to 20k samples dropped the gap to 8.67% and pushed validation accuracy to 87.58% without hurting training accuracy (96.24%). Unlike penalty regularizers that force underfitting, more data actually solves the root cause of variance.

* **Part 7 (Hyperparameter Tuning with 5-Fold CV & Final Evaluation):** Searched over learning rate, hidden layer width, and dropout rate across 12 configurations using 5-fold cross-validation. The winning setup was width 512, dropout 0.2, lr 0.0005 (CV score 85.11% +/- 0.51%). Retrained this configuration on the full 60,000 training set with BatchNorm and evaluated strictly once on the held-out 10,000 test set:
  * Test Accuracy: **90.45%**
  * Macro Precision: **90.67%**
  * Macro Recall: **90.45%**
  * Macro F1-Score: **90.52%**
  * Improvement over Part 2 Baseline: **+2.15 percentage points** (88.30% -> 90.45%). The wider network (512 units) preserved representational power for challenging classes like Shirts and Pullovers, while gentle dropout (0.2) and batch norm kept overfitting in check.
