---
layout: page
title: Masterface Attacks
description: Universal adversarial embeddings that impersonate 47% of identities in a 99.65%-accurate face verifier — showing that benchmark accuracy is not a security metric.
img: assets/img/Masterface-Attacks/fig_teaser.png
importance: 1
category: Published
---

<div class='container' style='background-color: #f8f9fa; max-width: 100%;
   padding: 20px 20px 5px 20px; margin-bottom: 20px;'>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Masterface-Attacks/fig_teaser.png" title="Same model, three pipelines: accuracy barely moves, masterface coverage swings by two orders of magnitude." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CosFace under three pipeline configurations. Pairwise accuracy stays ≈99.6%; masterface coverage swings from 47.2% to 0.23% — a more than 200× difference, invisible to the standard benchmark.
</div>
</div>

**Pairwise Accuracy Is Not Security: Masterface Attacks Expose a Structural Vulnerability in Face Verification.** Ehsan Nazari, 2026. [GitHub](https://github.com/enazari/Masterface-Attacks) · [Technical report (PDF)](/assets/pdf/masterface_technical_report.pdf)

Face verification powers phone unlock, online ID checks, and border control. A model embeds each face as a vector and accepts a match when two embeddings fall within a threshold distance. Benchmarks like LFW report pairwise accuracy — does the system tell same-identity pairs from different-identity pairs? Modern systems score above 99% and feel solved.

This project shows the benchmark and the security guarantee are not the same thing. **CosFace at 99.65% pairwise accuracy on LFW admits nine optimization-crafted embeddings that collectively impersonate 47.2% of LFW's 5,749 identities** — about 2,700 people, under the same threshold used for the accuracy headline. The pattern holds across ArcFace, AdaFace-IR101, AdaFace-ViT, and two FaceNet variants.

### Method

The attack is two phases:

- **Phase 1 — embedding-space search.** Given a set of identity embeddings on the unit hypersphere, search for a point whose τ-neighborhood covers as many of them as possible. The non-differentiable max-coverage objective is replaced with a smooth surrogate and optimized with **LM-MA-ES** (1,000 generations, population 100). Identity sets are partitioned into 9 clusters via spherical k-means, matching the budget of prior masterface work.
- **Phase 2 — image realization.** Given a masterface embedding, iteratively perturb a source face in pixel space (Adam, PGD-style perturbation budget) so its embedding under the face mapper approaches the target. The result is a plausible-looking face whose embedding lies within τ of many unrelated identities. **88–97% of the Phase-1 coverage survives realization.**

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Masterface-Attacks/phase2-arcface.png" title="Nine generated masterface images for ArcFace under MTCNN-DS + FAR≈0.001 from below. Collectively they cover 44.2% of all LFW identities." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Nine Phase-2 masterface images for ArcFace (MTCNN-DS, FAR≈0.001 under). Collectively they cover 44.2% of LFW identities.
</div>

The Phase-2 result reaches **49.957% coverage on FaceNet-CASIA**, surpassing the prior GAN-based result (Shmelkin et al., 2021, 43.82%) with a simpler, non-generative pipeline.

### Two pipeline "footguns"

Identical model, identical dataset; two implementation choices change attack surface by **up to two orders of magnitude** while pairwise accuracy moves by ≤0.1 pp:

- **FAR-threshold rounding direction** — rounding the operating threshold toward stricter vs. looser FAR.
- **Face-alignment strategy** — MTCNN-DavidSandberg vs. RetinaFace alignment.

Both are security-critical, both are structurally invisible to the standard benchmark.

### The irreducible floor (JT-Attack)

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Masterface-Attacks/j-attack-arcface.png" title="Joint-Threshold Attack against ArcFace under the strictest pipeline. Coverage (blue) far exceeds the 1/N theoretical maximum for a well-separated embedding space (red dashed)." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Joint-Threshold Attack against ArcFace under the safest pipeline (RetinaFace + FAR≈0.001 under). Coverage of N randomly chosen identities (blue) far exceeds the 1/N theoretical maximum for a well-separated embedding space (red dashed).
</div>

Under the *safest* pipeline configuration — the one where the broad masterface attack drops below 1% — a single embedding can still impersonate **any 2–4 randomly chosen identities ~100% of the time**, across every model tested. The theoretical maximum for a well-separated embedding space is 1/N; the observed curves blow past it. Pipeline mitigations bound the magnitude of broad attacks but leave targeted small-group attacks fully open. This is a model-level property, not an evaluation artifact.

### Takeaway

Pairwise verification benchmarks were never designed to expose security vulnerabilities, and this work makes the gap concrete. The same six face-verification models, evaluated on the same identities, look uniformly excellent on the benchmark and vary wildly under attack. If a face verification deployment matters for security, the pairwise accuracy headline is not the right number to report.

Code, configs, and the full technical report are on [GitHub](https://github.com/enazari/Masterface-Attacks).
