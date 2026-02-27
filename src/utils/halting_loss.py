import torch
import torch.nn.functional as F


def compute_trm_losses_and_halt(
    logits,
    targets,
    q_logits,
    correctness_threshold: float = 0.8,
    halt_threshold: float = 0.7,
    lambda_q: float = 0.1,
    ignore_index: int = -1,
):
    """
    Returns:
        total_loss: scalar tensor (CE + lambda_q * BCE), averaged over active tokens
        Halt_mask: [B] bool, to be used for the *next* supervision step
    """
    B = logits.size(0)
    q_halt_logits, q_correct_logits = torch.split(q_logits, 1, dim=-1)
    q_halt_logits = q_halt_logits.squeeze(-1)
    q_correct_logits = q_correct_logits.squeeze(-1)

    valid_token_mask = targets != ignore_index
    num_valid_tokens = valid_token_mask.sum(dim=-1)
    denom = num_valid_tokens.clamp(min=1)

    # ---- correctness label per sequence ---- #
    with torch.no_grad():
        pred_ids = logits.argmax(dim=-1)
        token_correct = (pred_ids == targets) & valid_token_mask
        correct_counts = token_correct.float().sum(dim=-1)
        frac_correct = correct_counts / denom
        # seq_gt_correct = (frac_correct > correctness_threshold).float()
        # target_halt = seq_gt_correct.view(B, 1).expand_as(q_halt_logits)
        # target_halt = frac_correct.view(B, 1).expand_as(q_halt_logits) 

    bce_per_token = F.binary_cross_entropy_with_logits(
        q_correct_logits, token_correct.float(), reduction="none"
    )
    bce_correct_loss = (bce_per_token * valid_token_mask.float()).sum() / (
        valid_token_mask.float().sum().clamp(min=1.0)
    )

    # bce_halt_per_token = F.binary_cross_entropy_with_logits(
    #     q_halt_logits, target_halt, reduction="none"
    # )
    # bce_halt_loss = (bce_halt_per_token * valid_token_mask.float()).sum() / (
    #     valid_token_mask.float().sum().clamp(min=1.0)
    # )
    q_halt_seq_logits = (q_halt_logits * valid_token_mask.float()).sum(dim=-1) / denom
    bce_halt_loss = F.binary_cross_entropy_with_logits(
        q_halt_seq_logits, frac_correct, reduction="mean"
    )

    # ---- halting decision for next supervision step ---- #
    with torch.no_grad():
        # q_halt_prob = torch.sigmoid(q_halt_logits)
        q_correct_prob = torch.sigmoid(q_correct_logits)

        # mean over non-pad tokens, for ACTIVE sequences only
        # q_halt_sum = (q_halt_prob * valid_token_mask.float()).sum(dim=-1)
        q_correct_sum = (q_correct_prob * valid_token_mask.float()).sum(dim=-1)
        # q_halt_mean = q_halt_sum / denom
        q_halt_mean = torch.sigmoid(q_halt_seq_logits)
        q_correct_mean = q_correct_sum / denom

        # model's own correctness decision:
        # "this sequence is correct enough"
        seq_model_correct = q_correct_mean > correctness_threshold

        # halting if:
        #  - still active,
        #  - model believes sequence is correct enough,
        #  - model believes halting probability is high enough
        # halt_now = (q_halt_mean > halt_threshold) & seq_model_correct
        halt_now = seq_model_correct

    return lambda_q * (bce_correct_loss + bce_halt_loss), halt_now, q_correct_mean
