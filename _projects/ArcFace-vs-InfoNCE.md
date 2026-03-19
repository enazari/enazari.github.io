---
layout: page
title: "Face Verification: ArcFace vs InfoNCE"
description: How far can contrastive loss get us compared to margin-based classification?
img: assets/img/ArcFace-vs-InfoNCE/infonce_pairs.png
importance: 1
category: Ongoing
---

<div class='container' style='background-color: #f8f9fa; max-width: 100%;
   padding: 20px 20px 5px 20px; margin-bottom: 20px;'>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ArcFace-vs-InfoNCE/infonce_pairs.png" title="InfoNCE pairs" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Can CLIP's contrastive loss alone learn to tell faces apart?
</div>
</div>

Most modern face verification systems are trained as classifiers — and then the classifier is thrown away. ArcFace learns a softmax head over identity classes, enforces angular margins, and discards it at inference. InfoNCE offers a more direct path: optimize embedding similarity itself, the same objective behind CLIP. This project puts them head-to-head on the same backbone and data.

## Results

All models trained on MS1M-ArcFace (85K identities), evaluated with 10-fold cross-validation at FAR@0.001.

| Experiment | LFW | CFP-FF | CFP-FP |
|-----------|-----|--------|--------|
| ArcFace (ResNet-50) | 99.72% | 99.80% | 95.84% |
| InfoNCE (ResNet-50) | 99.55% | 99.59% | 87.89% |

InfoNCE nearly matches ArcFace on frontal benchmarks but drops 8 points on CFP-FP (pose variation). I surmise this is largely a supervision asymmetry. Both losses are softmax cross-entropy — the difference is what goes in the denominator:

In ArcFace, each sample is pulled toward one positive class center and pushed away from all other 85,741 centers — fixed and always present. In InfoNCE, each sample is pulled toward one positive and pushed away from 1,448 other samples in the mini-batch — these change every iteration. CLIP uses 32,768-size batches; this suggests that increasing batch size may help narrow InfoNCE's performance gap on harder benchmarks such as CFP-FP.

However, InfoNCE's lack of fixed class centers may be a structural advantage. ArcFace embeds all faces on a hypersphere populated by 85,741 learned class centers — at inference, unseen faces tend to snap to the nearest center rather than occupying their own region of the space (see [Embedding space analysis](#embedding-space-analysis) below). InfoNCE has no such centers: the embedding space is shaped entirely by pairwise similarity, which may allow unseen identities to distribute more freely.

## Side experiments

### Does adding contrastive loss help ArcFace?

Hybrid loss (ArcFace + 0.5 × InfoNCE) yields no improvement — contrastive regularization adds nothing over ArcFace's margin alone.

| Experiment | LFW | CFP-FF | CFP-FP |
|-----------|-----|--------|--------|
| ArcFace (ResNet-50) | 99.72% | 99.80% | 95.84% |
| InfoNCE (ResNet-50) | 99.55% | 99.59% | 87.89% |
| **ArcFace + InfoNCE (ResNet-50)** | **99.72%** | **99.73%** | **95.30%** |

### Do foundation models transfer to faces?

Frozen CLIP, DINOv2, and I-JEPA perform poorly for identity discrimination (52–68% LFW), indicating that generic visual pretraining does not yield face-discriminative embeddings.

| Experiment | LFW | CFP-FF | CFP-FP |
|-----------|-----|--------|--------|
| ArcFace (ResNet-50) | 99.72% | 99.80% | 95.84% |
| InfoNCE (ResNet-50) | 99.55% | 99.59% | 87.89% |
| ArcFace + InfoNCE (ResNet-50) | 99.72% | 99.73% | 95.30% |
| CLIP ViT-B/32 (frozen) | 68.35% | 69.01% | 56.69% |
| DINOv2 ViT-B/14 (frozen) | 57.85% | 51.39% | 50.43% |
| I-JEPA ViT-H/14 (frozen) | 52.57% | 51.64% | 50.13% |

### LoRA adaptation of foundation models

LoRA (rank=8, <1% trainable params) with InfoNCE recovers foundation models to 89–96% LFW. DINOv2 leads on CFP-FF (96.81%), but all remain well below ResNet-50 trained from scratch on identity supervision.

| Experiment | LFW | CFP-FF | CFP-FP |
|-----------|-----|--------|--------|
| ArcFace (ResNet-50) | 99.72% | 99.80% | 95.84% |
| InfoNCE (ResNet-50) | 99.55% | 99.59% | 87.89% |
| ArcFace + InfoNCE (ResNet-50) | 99.72% | 99.73% | 95.30% |
| CLIP ViT-B/32 (frozen) | 68.35% | 69.01% | 56.69% |
| DINOv2 ViT-B/14 (frozen) | 57.85% | 51.39% | 50.43% |
| I-JEPA ViT-H/14 (frozen) | 52.57% | 51.64% | 50.13% |
| **CLIP ViT-B/32 + LoRA** | **96.08%** | **89.79%** | **72.24%** |
| **DINOv2 ViT-B/14 + LoRA** | **94.78%** | **96.81%** | **78.50%** |
| **I-JEPA ViT-H/14 + LoRA** | **89.12%** | **89.57%** | **72.73%** |

## Embedding space analysis

ArcFace trains a classifier over 85K identities and discards it at inference. How does it generalize to unseen faces?

ArcFace's final layer is a matrix of 85,741 L2-normalized class centers. During training, each embedding is pushed toward its own class center on the hypersphere. But when an unseen identity is fed through the frozen model, will it also land close to a class center?

To test this, I took positive pairs from LFW, CFP-FF, and CFP-FP and measured (a) cosine similarity between the two images in a pair, and (b) cosine similarity between each image and its nearest class center.

| Loss | Benchmark | Pair sim | Nearest center sim (A) | Nearest center sim (B) | % A closer to center | % B closer to center |
|------|-----------|----------|------------------------|------------------------|----------------------|----------------------|
| ArcFace | LFW | 0.714 | 0.630 | 0.631 | 43.3% | 44.8% |
| ArcFace | CFP-FF | 0.711 | 0.725 | 0.727 | 65.6% | 66.4% |
| ArcFace | CFP-FP | 0.515 | 0.726 | 0.573 | 90.2% | 77.1% |
| ArcFace+InfoNCE | LFW | 0.776 | 0.676 | 0.678 | 33.9% | 35.3% |
| ArcFace+InfoNCE | CFP-FF | 0.780 | 0.767 | 0.769 | 53.3% | 53.1% |
| ArcFace+InfoNCE | CFP-FP | 0.589 | 0.769 | 0.617 | 89.7% | 65.2% |

On LFW (easy, frontal), positive pairs are closer to each other than to any class center — generalization works as expected. On CFP-FP (pose variation), the picture flips: the frontal face snaps to a nearby class center (0.73) while the profile face drifts further (0.57), and pair similarity drops to 0.52. **90% of frontal faces are closer to a training class center than to their profile pair partner.** The model leans on the class-center geometry learned during training, which breaks under pose variation.

## Background

Face verification determines whether two face images belong to the same person — a pairwise similarity problem.

**FaceNet** (Schroff et al., CVPR 2015) was the first to frame it this way: triplet loss maps faces to a compact Euclidean space, and the embedding is used directly at inference. However, triplet mining is slow and unstable at scale.

**Margin-based losses** — SphereFace (Liu et al., CVPR 2017), CosFace (Wang et al., CVPR 2018), ArcFace (Deng et al., CVPR 2019) — reframe verification as multi-class classification during training. A softmax head over identity classes is added, angular margins enforce inter-class separation, and the classification head is discarded at inference. This indirect approach became the dominant paradigm.

**CoReFace** (Song & Wang, Pattern Recognition 2024) is the closest prior work. It adds contrastive regularization on top of margin-based classification to align training with pairwise evaluation. However, contrastive learning serves as a *regularizer* — the classification head remains the primary signal. This project asks: what happens when contrastive loss is the *only* signal?

## Future work

**Embedding space geometry.** Do InfoNCE embeddings truly avoid the class-center snapping observed with ArcFace? Do unseen identities distribute more freely across the hypersphere, or does a different kind of clustering emerge?

**Larger batch sizes.** The CFP-FP gap may largely be a supervision asymmetry. Scaling batch size is the most natural next step.

**Dataset overlap.** The embedding space analysis assumes MS1M-ArcFace and the evaluation benchmarks are identity-disjoint. Verifying this — and quantifying any overlap — is needed to validate those results.

## Experimental setting

- **Training data**: MS1M-ArcFace, 85,741 identities, 80/20 train/val split
- **Hardware**: 2× H100 GPUs, `accelerate` distributed, FP16 mixed precision
- **Evaluation**: LFW, CFP-FF, CFP-FP — 10-fold cross-validation at FAR@0.001
- **ResNet-50**: SGD (momentum 0.9, WD 5e-4), LR 0.1, 5 warmup epochs, 30 epochs, batch size 1,350–1,450
- **LoRA**: rank=8, α=8 on FFN layers; SGD LR 0.001, 15 epochs
- **Frozen baselines**: zero-shot evaluation only

Code: [github.com/enazari/ArcFace-vs-InfoNCE](https://github.com/enazari/ArcFace-vs-InfoNCE)
