# grpo-from-scratch

A minimal pure PyTorch implementation of GRPO-style RL training on GSM8K with LoRA over Qwen2.5-1.5B.

This repository explores a simple but interesting edge case: when each sampled trajectory is used for **only one policy update**, we do not need to keep a separate `π_old` model in memory. In this regime, the **scalar policy loss value** stays near zero, yet the **gradient remains informative**, and the model still learns.

In other words: this repo is both a compact GRPO implementation and a small empirical demonstration that **near-zero logged policy loss does not imply zero learning signal**.

## Main idea

In PPO/GRPO-style methods, one usually keeps an `π_old` policy to compute importance ratios when the same rollout is reused across multiple optimization steps.

Here, each sampled trajectory is used **exactly once**:

- generate a group of completions,
- compute rewards and normalized advantages,
- apply one update,
- discard the rollout.

Because of that, at update time we can set `π_old = π.detach()`. The importance ratio is then equal to 1 in value, so the scalar GRPO policy loss becomes zero. However, this does **not** mean the gradient is zero: differentiation still flows through the current policy log-probabilities, while the detached term only shifts the scalar objective. This saves memory and simplifies the implementation. 

There is a second memory-saving trick as well: in the LoRA setting, the reference policy `π_ref` is just the base model with adapters disabled. So there is no need to store a separate reference model either.

## What is implemented

- pure PyTorch GRPO-style training loop
- Qwen2.5-1.5B + LoRA
- GSM8K training
- GSM8K and GSM-Hard evaluation
- reward shaping based only on:
  - output format compliance,
  - producing a numeric answer,
  - producing the correct numeric answer

No supervised reasoning traces are used.

## Results

Pure RL training substantially improves math performance on GSM8K and also transfers part of the gain to GSM-Hard.

| Benchmark | Base model | After RL |
|---|---:|---:|
| GSM8K (strict) | ~30% | ~71% |
| GSM8K (soft) | ~56% | ~71% |
| GSM-Hard (strict) | ~13% | ~37% |
| GSM-Hard (soft) | ~27% | ~37% |

**Strict** means the model must follow the required output format exactly.  
**Soft** means only the final numeric answer is checked, ignoring exact formatting.

These results suggest that even this small-scale setup produces nontrivial learning and **some out-of-distribution transfer**.

## Prompt format

I use a simple prompt in the style of R1-like reasoning training, with a custom reasoning tag `<MYTHINK>` to make the learning effect easier to observe:

```python
x["prompt"] = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves "
    "it. The assistant first thinks about the reasoning process in the mind and then provides the "
    "user with the answer. The reasoning process and answer are enclosed within <MYTHINK>...</MYTHINK> "
    "and <answer>...</answer> tags, respectively, i.e.,\n"
    "<MYTHINK>\nreasoning process here\n</MYTHINK>\n"
    "<answer>\nanswer here\n</answer>\n"
    "User: " + x["question"] + "\nAssistant:"
)
```


## Checkpoints and logs

Training artifacts, checkpoints, and TensorBoard logs are available on [Hugging Face](https://huggingface.co/zinchse/Qwen2.5-1.5Bgrpo_gsm8k/tree/main).
