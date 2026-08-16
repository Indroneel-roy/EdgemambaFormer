# EdgeMambaFormer

**Edge-Guided Cross-Scale Mamba Transformer with Wavelet-Based Boundary Attention for Colonoscopy Polyp Segmentation**

Indroneel Roy, Mohammad Kamruzzaman Khan Prince — Shahjalal University of Science and Technology, Sylhet, Bangladesh

EdgeMambaFormer is a tri-hybrid CNN + Selective State-Space Model (Mamba) + Transformer architecture for polyp segmentation. A single model is trained once on pooled Kvasir-SEG + CVC-ClinicDB data and evaluated in-distribution on held-out splits of those two datasets, and zero-shot on CVC-ColonDB and ETIS-LaribPolypDB — no fine-tuning on the zero-shot sets.

## Architecture

```
Input (3×352×352)
     │
     ▼
PVTv2-B2 Encoder ──► f1, f2, f3, f4  (strides 4/8/16/32)
     │                    │
     │                    ▼
     │        Cross-Scale Mamba Module (CSMM)
     │        f2,f3,f4 → tokens → bidirectional
     │        selective scan → f2', f3', f4'
     │                    │
     ▼                    │
Wavelet Edge Attention Gate (WaveletEAG)
Haar DWT(f1) + gate(f4) → fe, boundary map σ
     │                    │
     └────────┬───────────┘
              ▼
  Dual-Branch Transformer Decoder
  local window attn (fe,f1) × global attn (f2',f3',f4')
  → cross-attention fusion → segmentation logit
```

Three components, each replacing a piece of a conventional encoder-decoder segmenter:

- **WaveletEAG** (`model/wavelet_eag.py`) — a fixed, non-learned single-level 2-D Haar DWT extracts multi-directional boundary signal (LH/HL/HH subbands) from the finest encoder stage, gated against deep semantic context via a learned sigmoid. Adds only **54,337 parameters** (<0.3% of the model) yet causes the **largest ablation drop** of the three components.
- **CSMM** (`model/csmm.py`) — tokenises the three coarsest encoder stages into one sequence and mixes them with a bidirectional selective-scan (S6/Mamba) block, propagating context across scales at linear cost instead of quadratic attention.
- **Dual-Branch Decoder** (`model/decoder.py`) — local windowed self-attention resolves fine boundary detail; global self-attention resolves multi-scale semantic context; a single cross-attention step fuses the two before the segmentation head.

## Results

### Table 1 — Shared-training protocol (train: 900 Kvasir-SEG + 550 CVC-ClinicDB)

mDice / mIoU (%) ↑. **Bold** = best per column. Baseline rows (U-Net, U-Net++, PraNet, SANet) as re-evaluated and reported by Polyp-PVT under this protocol; Polyp-PVT row is that paper's self-reported result.

| Method | Kvasir-SEG (in-dist.) | | CVC-ClinicDB (in-dist.) | | CVC-ColonDB (zero-shot) | | ETIS (zero-shot) | |
|---|---|---|---|---|---|---|---|---|
| | mDice | mIoU | mDice | mIoU | mDice | mIoU | mDice | mIoU |
| U-Net | 0.818 | 0.746 | 0.823 | 0.755 | 0.512 | 0.444 | 0.398 | 0.335 |
| U-Net++ | 0.821 | 0.743 | 0.794 | 0.729 | 0.483 | 0.410 | 0.401 | 0.344 |
| PraNet | 0.898 | 0.840 | 0.899 | 0.849 | 0.712 | 0.640 | 0.628 | 0.567 |
| SANet | 0.904 | 0.847 | 0.916 | 0.859 | 0.753 | 0.670 | 0.750 | 0.654 |
| Polyp-PVT | 0.917 | 0.864 | 0.937 | 0.889 | **0.808** | **0.727** | 0.787 | 0.706 |
| **EdgeMambaFormer (Ours)** | **0.9365** | **0.8899** | **0.9428** | **0.8952** | 0.7987 | 0.7142 | **0.7913** | **0.7064** |

EdgeMambaFormer attains the highest in-distribution accuracy of all compared methods (+1.95 mDice on Kvasir-SEG, +0.58 on CVC-ClinicDB vs. Polyp-PVT), and zero-shot accuracy competitive with the strongest baseline — matching or exceeding it on ETIS, trailing it by 0.9 points on CVC-ColonDB. We do not claim state-of-the-art on CVC-ColonDB specifically.

### Table 3 — Ablation study (Kvasir-SEG only)

| Configuration | mDice ↑ | mIoU ↑ | MAE ↓ |
|---|---|---|---|
| w/o WaveletEAG | 0.8302 | 0.7306 | 0.0475 |
| w/o CSMM | 0.9334 | 0.8816 | 0.0184 |
| w/o Dual-Branch Decoder | 0.9299 | 0.8761 | 0.0194 |
| **Full Model** | **0.9399** | **0.8924** | **0.0155** |

WaveletEAG causes a **−11.0 point mDice** drop when removed, despite contributing under 0.3% of the model's parameters — the largest single-component effect of the three, and the paper's central finding.

Full model: **25.43 M** parameters, **11.49 GFLOPs**, **76.13 ms** per-image inference latency (single NVIDIA Tesla T4).

## Pretrained Weights

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Indroneel/edgemambaformer)

The full-model checkpoint (25.43M params) reported in Table 3 is available on Hugging Face:
**[Indroneel/edgemambaformer](https://huggingface.co/Indroneel/edgemambaformer)**


## Repository structure

```
EdgeMambaFormer/
├── model/          # WaveletEAG, CSMM, decoder, full model, ablation variants
├── data/           # dataset discovery, pairing, augmentation
├── train/          # losses, metrics, train/eval loop
├── scripts/        # train.py, evaluate.py, ablate.py (CLI entry points)
├── notebooks/       # original Kaggle notebooks (train/eval + ablations)
├── assets/figures/  # architecture diagram, training curves, qualitative results

```

## Installation

```bash
git clone https://github.com/Indroneel-roy/EdgeMambaFormer.git
cd EdgeMambaFormer
pip install -r requirements.txt
```

## Datasets

Download the four public datasets and point the scripts at their roots:

| Dataset | Images | Role | Link |
|---|---|---|---|
| Kvasir-SEG | 1,000 | 900 train / 100 test (in-distribution) | [Kaggle](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) |
| CVC-ClinicDB | 612 | 550 train / 62 test (in-distribution) | [Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) |
| CVC-ColonDB | 380 | 100% zero-shot test | [Kaggle](https://www.kaggle.com/datasets/longvil/cvc-colondb) |
| ETIS-LaribPolypDB | 196 | 100% zero-shot test | [Kaggle](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb) |

## Quickstart

**Train the shared model:**
```bash
python scripts/train.py \
    --kvasir /path/to/Kvasir-SEG \
    --clinicdb /path/to/CVC-ClinicDB \
    --colondb /path/to/CVC-ColonDB \
    --etis /path/to/ETIS-LaribPolypDB \
    --epochs 100 --checkpoint ./checkpoints
```

**Evaluate a checkpoint on all four test sets:**
```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best_shared_edgemambaformer.pth \
    --kvasir /path/to/Kvasir-SEG \
    --clinicdb /path/to/CVC-ClinicDB \
    --colondb /path/to/CVC-ColonDB \
    --etis /path/to/ETIS-LaribPolypDB \
    --out results.json
```


## Training configuration

Adam, lr `1e-4`, weight decay `1e-4`, cosine annealing to `1e-6` over 100 epochs; batch size 16; image size 352×352; loss = weighted BCE + weighted IoU on both the main prediction head and the WaveletEAG auxiliary edge head (λ_pred=1.0, λ_edge=0.3). Augmentation: horizontal/vertical flip, 90° rotation, shift/scale/rotate, colour jitter, Gaussian noise. Full details in Section 3.6 of the paper.

## License

MIT — see [LICENSE](LICENSE).
