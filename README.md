# Model Architecture

Detailed explanation of the HybridViViT architecture used in this project.

## Table of Contents

- [Overview](#overview)
- [Architecture Components](#architecture-components)
- [Model 1: Multi-Task ViViT](#model-1-multi-task-vivit)
- [Model 2: Hybrid ResNet-Transformer](#model-2-hybrid-resnet-transformer)
- [Loss Function](#loss-function)
- [Data Flow](#data-flow)
- [Computational Complexity](#computational-complexity)

---

## Overview

The project implements two complementary deep learning models for spatio-temporal crop stress detection:

1. **Multi-Task Video Vision Transformer (ViViT)**: Pure transformer-based approach
2. **Hybrid ResNet-Transformer (HybridViViT)**: CNN-Transformer fusion (MAIN MODEL)

Both models perform dual tasks:
- **Classification**: Detect presence/absence of stress (Binary)
- **Regression**: Predict time-to-symptom (Continuous)

---

## Architecture Components

### 1. Spatial Feature Extraction

#### ResNet-18 Backbone
```
Input: (B, C, H, W) = Batch × 3 channels × 224 × 224 pixels
    ↓
Conv Layer 1: 3 → 64 filters (stride 2)
    ↓
ResBlock 1: 64 filters (2 sub-blocks)
    ↓
ResBlock 2: 128 filters (2 sub-blocks, stride 2)
    ↓
ResBlock 3: 256 filters (2 sub-blocks, stride 2)
    ↓
ResBlock 4: 512 filters (2 sub-blocks, stride 2)
    ↓
Output: (B, 512) feature vector
```

**Why ResNet-18?**
- Pre-trained on ImageNet (transfer learning)
- Efficient (only 18 layers)
- Proven for agricultural imaging
- Extracts rich spatial features

### 2. Temporal Modeling

#### Transformer Encoder
```
Input: (B, T, 512) = Batch × Time steps × Feature dimension
    ↓
[Linear Projection] → (B, T, 768)
    ↓
[Positional Encoding] (adds temporal position info)
    ↓
[Multi-Head Self-Attention Layer 1]
    • 6 attention heads
    • Each head: 128 dimensions
    • Learns temporal relationships
    ↓
[Feed-Forward Network]
    • Hidden: 2048 dimensions
    • Activation: ReLU
    ↓
[Layer Normalization + Residual Connection]
    ↓
[Repeat 3 times (3 transformer layers)]
    ↓
Output: (B, T, 768) transformed features
```

**Why Transformers?**
- Self-attention captures long-range temporal dependencies
- Parallelizable (efficient training)
- Learns what to focus on in time sequences
- No recurrence bottleneck (unlike RNNs)

### 3. Multi-Task Heads

#### Classification Head (Stress Detection)
```
Input: (B, T, 768)
    ↓
[Global Average Pooling] → (B, 768)
    ↓
[Dense Layer 1]: 768 → 256 (ReLU activation)
    ↓
[Dropout]: p=0.3
    ↓
[Dense Layer 2]: 256 → 64 (ReLU activation)
    ↓
[Dropout]: p=0.3
    ↓
[Dense Output]: 64 → 1 (no activation)
    ↓
Output: Logit (passed to BCEWithLogitsLoss)
Final output: Sigmoid(logit) ∈ [0, 1] (probability)
```

#### Regression Head (Time Prediction)
```
Input: (B, T, 768)
    ↓
[Global Average Pooling] → (B, 768)
    ↓
[Dense Layer 1]: 768 → 256 (ReLU activation)
    ↓
[Dropout]: p=0.3
    ↓
[Dense Layer 2]: 256 → 64 (ReLU activation)
    ↓
[Dropout]: p=0.3
    ↓
[Dense Output]: 64 → 1 (linear)
    ↓
Output: Time value ∈ [0, 5] days
```

---

## Model 1: Multi-Task ViViT

### Architecture Diagram
```
Input Sequence: (B, T, H, W, C)
(Batch, Time, Height, Width, Channels)
        ↓
[3D Patch Embedding]
Spatial-Temporal patch extraction
        ↓
[Transformer Encoder - 4 layers, 8 heads]
        ↓
[Shared Representation]
        ↙              ↘
[Classification] [Regression]
Head 1             Head 2
```

### Key Parameters
- Input shape: (B=4, T=10, H=128, W=128, C=3)
- Patch size: (2, 16, 16) → 40 patches per sample
- Hidden dimension: 128
- Attention heads: 8
- Encoder layers: 4

### Advantages
- Pure vision transformer approach
- Efficient spatial-temporal processing
- End-to-end learnable embeddings

### Limitations
- Requires more training data
- Higher computational cost
- Slower convergence

---

## Model 2: Hybrid ResNet-Transformer (HybridViViT)

### Architecture Diagram
```
Input Sequence: T frames × (H, W, 3)
        ↓
[ResNet-18 Feature Extraction]
Process each frame independently
        ↓
Feature Sequence: (B, T, 512)
        ↓
[Transformer Encoder - 3 layers, 6 heads]
Learn temporal dynamics
        ↓
[Pooled Representation]
        ↓
[Shared Feature Vector]
        ↙              ↘
[Classification] [Regression]
Head             Head
```

### Key Parameters
- ResNet-18: Pre-trained on ImageNet
- Feature dimension: 512
- Transformer layers: 3
- Attention heads: 6
- Hidden dimension: 768
- Dropout: 0.3

### Advantages
- **Transfer Learning**: Pre-trained ResNet backbone
- **Efficiency**: Less training required
- **Stability**: Proven CNN-Transformer combination
- **Convergence**: Faster training (our main model!)

### Disadvantages
- Two-stage feature extraction (less end-to-end)
- Fixed spatial features (ResNet not fine-tuned in early training)

---

## Loss Function

### Joint Multi-Task Loss

```
L_total = L_cls + λ * L_reg

Where:
  L_cls = BCEWithLogitsLoss (Classification)
  L_reg = HuberLoss (Regression)
  λ = 1.2 (weight factor)
```

### Loss Components

**Classification Loss (BCEWithLogitsLoss)**
```
L_cls = -[y*log(σ(z)) + (1-y)*log(1-σ(z))]

Where:
  y = Ground truth (0 or 1)
  z = Model logit
  σ(z) = Sigmoid activation
```

**Regression Loss (HuberLoss)**
```
L_reg = {
  0.5 * r² if |r| ≤ δ
  δ * (|r| - 0.5*δ) if |r| > δ
}

Where:
  r = y - ŷ (prediction error)
  δ = 1.0 (delta parameter)
```

**Why These Losses?**
- BCEWithLogitsLoss: Numerically stable for binary classification
- HuberLoss: Robust to outliers in regression
- Joint training: Improve both tasks simultaneously

---

## Data Flow

### Training Forward Pass

```
Step 1: Input Batch
├─ Shape: (32, 10, 128, 128, 3)
├─ 32 samples
├─ 10 time steps
└─ 128×128 RGB images

Step 2: Feature Extraction (ResNet-18)
├─ Process each frame: (32*10, 3, 128, 128) → (32*10, 512)
├─ Reshape: (32, 10, 512)
└─ Output: Feature sequence

Step 3: Temporal Modeling (Transformer)
├─ Input: (32, 10, 512)
├─ Attention heads: 6
├─ Output: (32, 10, 768)
└─ Pooling: (32, 768)

Step 4: Task Heads
├─ Classification: (32, 768) → (32, 1) → Sigmoid → (32,)
├─ Regression: (32, 768) → (32, 1) → Clip to [0,5] → (32,)
└─ Outputs: (stress_prob, time_pred)

Step 5: Loss Computation
├─ L_cls = BCEWithLogitsLoss(logits, targets_cls)
├─ L_reg = HuberLoss(pred_time, targets_reg)
├─ L_total = L_cls + 1.2 * L_reg
└─ Loss scalar

Step 6: Backward Pass
├─ Gradients computed
├─ Parameters updated via AdamW
└─ One training step complete
```

---

## Computational Complexity

### Model Size
```
ResNet-18:         11.7M parameters
Transformer:       ~5.2M parameters
Task Heads:        ~1.1M parameters
────────────────────────────────
Total:             ~18M parameters

Memory per sample: ~200 MB (with gradients)
Batch size 32:     ~6.4 GB GPU memory
```

### Training Speed
```
Time per epoch:     ~5 minutes (GPU)
5 epochs total:     ~25 minutes
Dataset size:       1,218 samples
Samples/second:     ~81 (with GPU)
```

### Inference Speed
```
Per sample:         ~50 ms
Per batch (32):     ~150 ms
Throughput:         ~210 samples/second
```

---

## Architecture Variants

### Lightweight Version (For Mobile)
```
ResNet-18 → Replace with MobileNet v2
Transformer → 2 layers instead of 3
Feature dim → 256 instead of 512
Result: 3M parameters (6x smaller)
```

### High-Performance Version
```
ResNet-50 → Larger backbone
Transformer → 6 layers instead of 3
Feature dim → 1024 instead of 512
Attention heads → 12 instead of 6
Result: 120M parameters (best accuracy)
```

---

## Key Design Decisions

1. **ResNet-18 over ResNet-50**: Balance speed & accuracy
2. **3 Transformer layers**: Empirically optimal for this dataset
3. **6 attention heads**: Standard for 768-dim features
4. **BCEWithLogitsLoss**: Numerically stable
5. **HuberLoss**: Robust to time-to-symptom outliers
6. **λ=1.2**: Empirically tuned weight balance
7. **Dropout 0.3**: Prevent overfitting on small dataset

---

## Future Improvements

- [ ] Add spatial attention mechanisms
- [ ] Implement multi-scale temporal modeling
- [ ] Try Vision Transformer (ViT) backbone instead of ResNet
- [ ] Add uncertainty quantification
- [ ] Implement knowledge distillation for mobile deployment

---

**Last Updated**: June 2024  
**Version**: 1.0.0
