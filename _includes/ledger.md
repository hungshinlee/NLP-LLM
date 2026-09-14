<!-- 此檔由 scripts/build_weeks.py 自動產生，請勿直接編輯；請改 docs/course-outline.md 後重跑腳本。 -->

## The Compute Ledger: The Course's Shared Coordinate System

Hand this out in week 1, then fill in each layer's contribution as the course goes. It is the device that lets an abstract argument fall back onto a number at any point.

```
Training                                   Inference (per token)
─────────────────────────                  ─────────────────────────
C_train ≈ 6 · N · D  FLOPs                 prefill:  ≈ 2 · N · L_in  FLOPs   (compute bound)
  N = parameters                           decode:   ≈ 2 · N        FLOPs   (memory bound)
  D = training tokens                      KV cache: 2 · n_layer · n_kv_head · d_head
  6 = 2 forward + 4 backward                         · L · bytes_per_elem
                                           → what bounds decode is moving N and the KV
MoE: N_total sets memory,                    from HBM into SRAM, not the arithmetic
     N_active sets FLOPs
```

Three numbers students should work out for themselves in the first session, and revisit as the course goes:

1. The training FLOPs for a 7B dense model at Chinchilla-optimal ($D \approx 20N$), and how many years that is on a single T4.
2. The KV cache for that model at batch 1, 32k context, FP16 — in GB, against the T4's 16 GB. **That number is the shared motivation for W5, W6 and W11.**
3. Working backwards from the activation ratio: why a 2026 flagship is rational at roughly a trillion total parameters and tens of billions active, and where that design moves the cost *to*.
