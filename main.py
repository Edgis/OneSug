from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

NUM_NEGS_ = 4
PADDING_VALUE_ = 0
LABEL_PAD_TOKEN_ID = -100
TRAIN_PARAMS = {
    'beta': 0.5,
    'sft_loss_weight': 0.4,
    'delta': 0.5
}

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def listwise_dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: Dict[str, torch.FloatTensor],
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: Dict[str, torch.FloatTensor],
        chosen_rw_score: torch.FloatTensor,
        rejected_rw_score: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Sorry, we cannot release the listwise_dpo_loss function due to the company's privacy policy.
        """
        losses, chosen_rewards, rejected_rewards = None, None, None
        return losses, chosen_rewards, rejected_rewards

def concatenated_inputs(batch):
    concatenated_batch = {}
    max_length = max(batch["chosen_labels"].shape[1], max([batch[f"rejected_labels_{i}"].shape[1] for i in range(NUM_NEGS_)]))
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor) and 'score' not in k:
            pad_value = LABEL_PAD_TOKEN_ID if "labels" in k else PADDING_VALUE_
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        for i in range(NUM_NEGS_):
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor) and 'score' not in k and k.endswith(f'_{i}'):
                pad_value = LABEL_PAD_TOKEN_ID if "labels" in k else PADDING_VALUE_
                concatenated_key = k.replace("rejected", "concatenated").replace(f'_{i}', '')
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )
    concatenated_batch["concatenated_input_ids"] = pad_to_length(batch["prompt_input_ids"].repeat(NUM_NEGS_ + 1, 1), max_length, pad_value=0)
    concatenated_batch["concatenated_attention_mask"] = pad_to_length(batch["prompt_attention_mask"].repeat(NUM_NEGS_ + 1, 1), max_length, pad_value=0)
    return concatenated_batch

def concatenated_forward(model, batch):
    def _get_batch_logps(
                logits: torch.FloatTensor,
                labels: torch.LongTensor,
                average_log_prob: bool = False
            ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")    
        loss_mask = labels != LABEL_PAD_TOKEN_ID
        labels[labels == LABEL_PAD_TOKEN_ID] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    concatenated_batch = concatenated_inputs(batch)
    len_chosen = batch["chosen_labels"].shape[0]
    with autocast():
        _, all_logits, *_ = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            decoder_input_ids=concatenated_batch["concatenated_decoder_input_ids"],
            decoder_attention_mask=concatenated_batch["concatenated_decoder_attention_mask"],
            labels=concatenated_batch["concatenated_labels"]
        )
    all_logps = _get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        False,
    )
    chosen_logps = all_logps[:len_chosen]
    chosen_logits = all_logits[:len_chosen]
    rejected_logps, rejected_logits = {}, {}
    for i in range(NUM_NEGS_):
        rejected_logps[i] = all_logps[len_chosen*(i+1) : len_chosen*(i+2)]
        rejected_logits[i] = all_logits[len_chosen*(i+1) : len_chosen*(i+2)]
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

def get_batch_metrics(model, ref_model, batch):
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
    ) = concatenated_forward(model, batch)
    losses = None
    (
        reference_chosen_logps,
        reference_rejected_logps,
        _,
        _,
    ) = concatenated_forward(ref_model, batch)

    chosen_rw_score = batch['chosen_rw_score'].view(-1)
    rejected_rw_score = batch['rejected_rw_score'].view(-1)
    chosen_rw_score = batch['chosen_rw_score'].view(-1)
    rejected_rw_score = batch['rejected_rw_score'].view(-1)
    losses, _, _ = listwise_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        chosen_rw_score,
        rejected_rw_score)
    return losses.mean()

def main(policy_model, reference_model, train_loader):
    for data in train_loader:
        policy_model.train()
        loss = get_batch_metrics(policy_model, reference_model, data)
        loss.backward()
