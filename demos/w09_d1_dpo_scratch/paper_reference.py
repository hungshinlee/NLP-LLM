# -*- coding: utf-8 -*-
# 此檔由 bin/publish_demos.py 從 private repo 的 <course>/demos/ 複製而來，請勿在公開 repo 直接編輯。
"""測試 (v) 的對照：照 DPO 原論文附錄 B 的寫法（Rafailov et al., 2023, arXiv:2305.18290, Appendix B）在 PyTorch 上重打一次。

與 dpo_live.py 的差別在介面與括號的位置，數學相同：
- 介面：論文把一批回答的 log 機率放在一個張量裡，另給兩個索引張量指出哪些是 y_w、哪些是 y_l。
- 括號：論文先算 policy 的 (y_w − y_l)、再算 reference 的 (y_w − y_l)、兩者相減；
  課堂上寫的是先算 y_w 的 (policy − reference)、再算 y_l 的、兩者相減。
論文的函數回傳每一對的 loss（不取平均）與每段回答的隱含 reward；這裡也一樣。
"""
from __future__ import annotations


def dpo_loss_paper(policy_logps, reference_logps, w_index, l_index, beta):
    import torch.nn.functional as F
    policy_gap = policy_logps[w_index] - policy_logps[l_index]
    reference_gap = reference_logps[w_index] - reference_logps[l_index]
    per_pair = -F.logsigmoid(beta * (policy_gap - reference_gap))
    implied_reward = beta * (policy_logps - reference_logps).detach()
    return per_pair, implied_reward
