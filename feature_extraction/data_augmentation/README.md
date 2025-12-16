# Structure-Aware Cross-Modal Augmentation

This module implements the **Structure-Aware Cross-Modal Augmentation (CMA)** strategy proposed in **STAR**, which is designed to improve the robustness and generalization of zero-shot website fingerprinting under HTTPS.

Unlike conventional data augmentation methods that operate independently on each modality, CMA **jointly augments the logic modality and the traffic modality in a structurally consistent manner**, preserving cross-modal semantic alignment.

---

## Motivation

In real-world web environments, websites evolve continuously:
- resources may be added, removed, or migrated,
- third-party services may change,
- CDN or server-side deployments may fluctuate.

These changes often affect **only part of a website**, rather than the entire page.  
However, naive data augmentation strategies (e.g., random cropping or noise injection) may:
- destroy meaningful semantic–traffic correspondences,
- introduce unrealistic cross-modal pairs,
- harm contrastive alignment training.

**Structure-Aware Cross-Modal Augmentation** addresses this problem by:
- simulating *partial website evolution*,
- while strictly maintaining **semantic consistency between modalities**.

---

## Core Idea

The key insight is that **server IP addresses act as a natural structural anchor shared by both modalities**:

- In the **logic modality**, resources can be grouped by their server IP.
- In the **traffic modality**, packets can be associated with the same IP via flow metadata.

CMA performs **IP-level structural deletion**, meaning:

> If a server (IP) is removed from the logic view, *all corresponding traffic packets from that server are also removed*.

This ensures that the augmented pair remains **internally coherent and semantically valid**.

---

## Augmentation Procedure

Given a paired sample:

- Logic modality:  
  `R = { r₁, r₂, ..., rₙ }`, each resource annotated with a server IP
- Traffic modality:  
  `P = { p₁, p₂, ..., pₘ }`, each packet annotated with a server IP

The augmentation proceeds as follows:

1. **Group resources by server IP**  
   - Construct resource groups `G(s)` for each server `s`.

2. **Weighted IP sampling**  
   - Servers hosting fewer resources are sampled with higher probability:
     ```
     w(s) = 1 − |G(s)| / |R|
     ```
   - This avoids always deleting dominant servers and encourages diverse substructures.

3. **Stochastic deletion budget**  
   - Sample a deletion threshold from a Gaussian prior:
     ```
     T ~ N(μ = 0.3, σ = 0.1) × |R|
     ```
   - This introduces controlled randomness in augmentation strength.

4. **Synchronized cross-modal deletion**
   - For each sampled server IP `s`:
     - Remove all resources `G(s)` from the logic modality
     - Remove all packets associated with `s` from the traffic modality
   - Continue until the deletion budget is reached.

5. **Return an augmented logic–traffic sub-pair**

---

## Why This Work

