# 🔍🧠 HSRIS — Hybrid Semantic Retrieval and Intelligence System

> **Assignment 3 — Natural Language Processing Pipeline**  
> Built entirely from scratch using PyTorch — No sklearn used

---

## 👤 Student Information

| Field | Details |
|---|---|
| **Name** | Ahmad Abdullah |
| **Roll Number** | 23F-3027 |
| **Course** | Data Science |
| **Assignment** | DS_ASS03_23F_3027 |

---

## 🌐 Live Demo and Links

| Resource | Link |
|---|---|
| 🚀 **Live Gradio App** | https://f347f7b8007dcaf286.gradio.live |
| 📓 **Kaggle Notebook** | https://www.kaggle.com/code/f233027ahmadabdullah/23f-3027-ds-ass3 |
| 📝 **Medium Article** | https://medium.com/@ahmadisonlyone/c13524a7db2b |

---

## 📌 Project Overview

HSRIS is a complete multi-stage NLP pipeline that searches 8,469 customer support tickets using a hybrid approach combining statistical keyword matching (TF-IDF) with neural semantic understanding (GloVe embeddings). The system finds similar past tickets for any new incoming support query.

The key innovation is the hybrid scoring formula:
```
FinalScore = α × TF-IDF_Score + (1 - α) × GloVe_Score
```

Where α = 0.4 by default, giving 40% weight to keyword matching and 60% weight to semantic meaning.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Dataset** | Customer Support Ticket Dataset |
| **Total Records** | 8,469 tickets |
| **Ticket Types** | Technical issue, Billing inquiry, Refund request, Product inquiry, Cancellation request |
| **Priority Levels** | Critical, High, Medium, Low |
| **Channels** | Email, Phone, Social media, Chat |
| **Source** | Kaggle — waseemalastal/customer-support-ticket-dataset |

---

## 🏗️ System Architecture
```
Raw Ticket Text
      │
      ▼
┌─────────────────┐
│   Tokenizer     │  Regex cleaning, stopword removal,
│   + N-grams     │  bigrams and trigrams generation
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌────────┐
│TF-IDF │  │ GloVe  │
│Sparse │  │ 300-dim│
│Engine │  │Semantic│
└───┬───┘  └───┬────┘
    │           │
    └─────┬─────┘
          │
          ▼
   Hybrid Score
   0.4×TF-IDF + 0.6×GloVe
          │
          ▼
   Top-K Results
```

---

## ⚙️ Implementation Phases

### Phase 1 — Environment Setup and Data Loading
- Loaded dataset from Kaggle input directory
- Verified dual Tesla T4 GPU setup (14.56 GB each)
- Explored all 17 columns, cleaned data
- Extracted 5 focus fields for NLP pipeline

### Phase 2 — Categorical Encoders (No sklearn)
- Built `CustomLabelEncoder` from scratch using Python dicts
- Built `CustomOneHotEncoder` from scratch using NumPy
- Handles unseen categories safely without crashing
- Converted to PyTorch tensors on GPU
```python
# Label Encoding result
{'Critical': 0, 'High': 1, 'Low': 2, 'Medium': 3}

# One-Hot Encoding result
['Chat', 'Email', 'Phone', 'Social media'] → binary vectors
```

### Phase 3 — Custom Tokenizer and TF-IDF Engine
- Regex tokenizer with lowercase, punctuation removal, stopwords
- Bigram and trigram generation using sliding window
- Top 5,000 vocabulary from combined Subject + Description
- Sparse BoW tensor using `torch.sparse_coo_tensor`
- Manual TF-IDF computation using PyTorch tensor operations
```
Vocabulary size:     5,000 tokens
Total unique tokens: 138,779 (with n-grams)
Sparse density:      1.33% (98.67% memory savings)
Build time:          0.44 seconds
```

### Phase 4 — GloVe Semantic Embeddings
- Downloaded GloVe 6B 300-dimensional pretrained vectors
- Loaded 400,000 word vectors into frozen PyTorch Embedding layer
- OOV tokens handled with zero vectors
- TF-IDF weighted sentence averaging (not simple mean)
- Built all 8,469 sentence vectors in 3.35 seconds
```
GloVe vocabulary:  400,000 words
Dataset coverage:  68.00% (5,577 words found)
OOV words:         2,625 (zero vector fallback)
Embedding dims:    300
Build time:        3.35 seconds
```

### Phase 5 — Hybrid Search and Dual GPU Optimization
- Combined TF-IDF and GloVe scores with alpha weighting
- Implemented `torch.nn.DataParallel` across both T4 GPUs
- Batch processed 100 test queries for benchmarking
```
Batch Size 10  → 0.082s → 121 queries/second
Batch Size 25  → 0.084s → 296 queries/second
Batch Size 50  → 0.143s → 350 queries/second
Batch Size 100 → 0.268s → 374 queries/second
```

### Phase 6 — Evaluation and Qualitative Analysis
- Precision@5 evaluated on 200 sampled tickets
- 3-way comparison: TF-IDF vs GloVe vs Hybrid
- Found 5 qualitative examples where GloVe beats TF-IDF
- Generated side-by-side comparison visualizations
```
TF-IDF Only (α=1.0):  Mean P@5 = 0.3690
GloVe Only  (α=0.0):  Mean P@5 = 0.3610
Hybrid      (α=0.4):  Mean P@5 = 0.3670
```

### Phase 7 — Gradio Dashboard Deployment
- Professional dark-themed Gradio 5.50.0 web application
- Alpha slider (0.0 to 1.0) with real-time mode label
- Predicted ticket type with confidence percentage
- Top 3 similar tickets with visual score bars
- 4 quick example query buttons
- Deployed with public shareable URL

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Total Tickets | 8,469 |
| Vocabulary Size | 5,000 |
| GloVe Dimensions | 300 |
| GloVe Coverage | 68.00% |
| Sparse Density | 1.33% |
| Mean P@5 (Hybrid) | 0.3670 |
| Mean P@5 (TF-IDF) | 0.3690 |
| Mean P@5 (GloVe) | 0.3610 |
| Max Throughput | 374 queries/second |
| GPU Setup | Dual Tesla T4 (14.56 GB each) |

---

## 🔬 Key Technical Decisions

**Why TF-IDF weighted GloVe averaging?**
Simple mean pooling treats all words equally. Rare important words like "billing" or "kernel" get diluted by common words like "the" or "please". TF-IDF weighting ensures important terms contribute more to the sentence vector.

**Why combine Subject and Description?**
Ticket subjects like "Network problem", "Account access", "Data loss" are concise problem summaries. Including them in tokenization improves vocabulary quality and search accuracy.

**Why alpha = 0.4?**
GloVe captures semantic similarity that TF-IDF misses entirely. Giving 60% weight to semantic understanding produces more robust results across diverse query phrasings.

**Why sparse tensors?**
The BoW matrix (8469 × 5000) has only 1.33% non-zero entries. Dense storage would use ~170MB unnecessarily. Sparse tensor uses only the non-zero values saving significant memory.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Word Embeddings | GloVe 6B 300d |
| Web App | Gradio 5.50.0 |
| GPU Platform | Kaggle Dual Tesla T4 |
| Visualization | Matplotlib |

---

## 🚫 Constraints Met

- ✅ No sklearn used anywhere
- ✅ No scipy used
- ✅ No high-level NLP wrappers
- ✅ All encoders built from scratch
- ✅ TF-IDF computed manually
- ✅ Sparse tensors for memory efficiency
- ✅ Dual GPU DataParallel implemented
- ✅ TF-IDF weighted averaging (not simple mean)

---

## 📁 Repository Structure
```
HSRIS-NLP-Assignment/
│
├── DS_ASS01_23F_3027.ipynb    ← Main notebook (all 7 phases)
└── README.md                  ← This file
```

---

## 📞 Contact

**Ahmad Abdullah**  
Roll Number: 23F-3027  
GitHub: https://github.com/AhmadAbdullah75
Live Demo (Hugging Face): https://ahmadabdullah27-hsris-nlp-assignment.hf.space
---

*Built with PyTorch + GloVe + TF-IDF | Assignment 3 | Dual T4 GPU | Gradio 5.50.0*
```
