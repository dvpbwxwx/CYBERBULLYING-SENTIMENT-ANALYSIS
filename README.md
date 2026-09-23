# 🛡️ Comparative Analysis of Automatic Labeling with Cohen's Kappa Validation for Cyberbullying on TikTok Using Pre-Trained Transformers

An end-to-end Machine Learning & NLP research project evaluating automatic labeling techniques (Zero-Shot Classification vs. InSet Lexicon) validated with Cohen's Kappa Score, and fine-tuning Pre-trained Transformers (IndoBERT vs. RoBERTa) to detect cyberbullying in informal Indonesian TikTok comments.

---

## 📌 Research Overview & Highlights
* **Dataset:** 5,864 TikTok comments scraped using Apify (TikTok Comments Scraper).
* **Gold Standard Validation:** 1,141 human-annotated comments used as ground truth for Cohen's Kappa coefficient testing.
* **Key Finding:** **Zero-Shot Classification** achieved **Cohen's Kappa of 0.9132** (Almost Perfect Agreement), vastly outperforming **InSet Lexicon** which scored **0.2585** (Fair Agreement) due to false positives on casual Indonesian slang.
* **Best Model:** **IndoBERT trained on Zero-Shot labels** reached the highest performance with **91.10% Accuracy & F1-Score** and an **AUC ROC of 0.96**.
* **Publication:** Published in *Journal of Applied Informatics and Computing (JAIC)*, Vol. 10, No. 4, August 2026.

---

## 🛠️ Tech Stack & Methodology
* **Language & Frameworks:** Python, PyTorch, Hugging Face Transformers, Scikit-Learn, Pandas, NumPy.
* **Preprocessing:** Cleaning, Case Folding, Character Normalization, and Slang Normalization (No stemming/stopwords applied for optimal Transformer context retention).
* **Labeling Approaches:**
  * **Zero-Shot Classification:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.
  * **Lexicon-Based:** InSet Lexicon (Koto et al.).
* **Imbalance Handling:** Random Oversampling (ROS) applied to Zero-Shot labels (yielding 8,198 balanced dataset points split 80:10:10 stratified ratio).
* **Fine-Tuned Architectures:**
  * `indobenchmark/indobert-base-p1` (WordPiece Tokenizer).
  * `w11wo/indonesian-roberta-base-sentiment-classifier` (Byte-Pair Encoding).

---

## 📊 Performance Comparison

| Model Architecture | Labeling Method | Accuracy | Precision | Recall | F1-Score | AUC ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IndoBERT** | **Zero-Shot (mDeBERTa)** | **91.10%** | **91.10%** | **91.10%** | **91.10%** | **0.96** |
| RoBERTa | Zero-Shot (mDeBERTa) | 85.85% | 86.07% | 85.85% | 85.83% | 0.93 |
| IndoBERT | InSet Lexicon | 87.39% | 86.94% | 86.94% | 86.94% | 0.94 |
| RoBERTa | InSet Lexicon | 81.09% | 80.61% | 79.92% | 80.21% | 0.88 |

---

## ⚙️ Hyperparameters Configuration
* **Learning Rate:** 1e-5
* **Batch Size:** 16
* **Epochs:** Max 10 (Early Stopping with patience 2)
* **Dropout Rate:** 0.3 | **Weight Decay:** 0.05 | **Label Smoothing:** 0.1
