import os
import gc
import torch
import wandb
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import torch.nn as nn
from utils import Utility


class Trainer(metaclass=ABCMeta):
    def __init__(self, save_root, accel):
        self.save_root = save_root
        self.accel = accel

    @abstractmethod
    def compute_loss_and_update(self, *args, **kwargs):
        pass
    
    def memory_optimization(self):

        # wait
        self.accel.wait_for_everyone()

        # memory deallocation
        gc.collect()

        # removing cache
        torch.cuda.empty_cache()

        # wait
        self.accel.wait_for_everyone()

    def wait_and_save_ckpt(self, **kwargs):

        model = kwargs['model']
        batch_ind = kwargs['batch_ind']
        length_dataloader = kwargs['epochs'] * len(kwargs['train_dataloader'])
        save_number = kwargs['save_number']
        processor = kwargs['processor']

        # wait for everyone
        self.accel.wait_for_everyone()
        if batch_ind+1 in [int(i/save_number*length_dataloader) for i in range(1, save_number+1)]:
            
            # Student
            unwrapped_model = self.accel.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(self.save_root, f'{batch_ind+1}'),
                is_main_process=self.accel.is_main_process and self.accel.local_process_index==0,
                save_function=self.accel.save,
                state_dict=self.accel.get_state_dict(model),
                max_shard_size='3GB'
                )

            # processor
            processor.save_pretrained(os.path.join(self.save_root, f'{batch_ind+1}'))
            
            # print
            self.accel.print(f"----{batch_ind+1}: Save Comleted!!----")
        
        # wait for everyone
        self.accel.wait_for_everyone()

    def train(self, **kwargs):
        """
        necessary kwargs

        - model
        - vllm_model
        - epochs
        - train_dataloader
        - optimizer
        - scheduler
        - processor
        - max_new_tokens
        - wandb
        - save_number
        """
        for epoch in range(kwargs['epochs']):

            # progress bar
            prog_bar = tqdm(enumerate(kwargs['train_dataloader']),
                            disable=not (self.accel.is_main_process and self.accel.local_process_index==0),
                            total=len(kwargs['train_dataloader']))

            # training start
            for batch_ind, inputs in prog_bar:

                # memory opt
                self.memory_optimization()

                # forward & backward
                with self.accel.accumulate(kwargs['model']):
                    # backwarding loss with gradient accumulation
                    loss_dict = self.compute_loss_and_update(inputs, **kwargs)
                
                # wandb logging
                if kwargs['wandb'] and self.accel.is_main_process and self.accel.local_process_index==0:
                    update_wandb_dict = {'lr': kwargs['scheduler'].get_last_lr()[0]}
                    for k, v in loss_dict.items(): update_wandb_dict.update({k: v})
                    wandb.log(update_wandb_dict)

                # displaying progress bar
                GPU0_usage = torch.cuda.memory_reserved(device=0) / 1024**3
                prog_bar.set_description(f"[GPU0:{GPU0_usage:.0f}][LR:{kwargs['scheduler'].get_last_lr()[0]:.6f}] " +\
                                            " | ".join([f"{k}: {v:.3f}" for k, v in loss_dict.items()]), refresh=True)

                # saving the model
                kwargs['batch_ind'] = epoch * len(kwargs['train_dataloader']) + batch_ind
                self.wait_and_save_ckpt(**kwargs)


class GRPOTrainer(Trainer):
    def __init__(self, save_root, accel):
        # super init
        super().__init__(save_root, accel)

    def compute_loss_and_update(self, inputs, **kwargs):
        """
        Arguments:
        - inputs
        - model
        - ref_model
        - vllm_model
        - vllm_sampling_params
        - processor
        - optimizer
        - scheduler
        - num_gens
        - grpo_iters
        - clip_high_eps
        - clip_low_eps
        - kld_beta
        """
        model = kwargs["model"]
        ref_model = kwargs["ref_model"]
        vllm_model = kwargs["vllm_model"]
        vllm_sampling_params = kwargs["vllm_sampling_params"]
        processor = kwargs["processor"]
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["scheduler"]
        num_gens = kwargs["num_gens"]
        max_new_tokens = kwargs['max_new_tokens']
        temperature = kwargs['temperature']
        grpo_iters = kwargs["grpo_iters"]
        clip_high_eps = kwargs["clip_high_eps"]
        clip_low_eps = kwargs["clip_low_eps"]
        kld_beta = kwargs["kld_beta"]

        # preprocessing for text and image
        prompt_list, image_list, qa_index_list = Utility.preprocess(inputs, processor)

        # vLLM generation and merging
        completion_texts = Utility.vLLM_generation(vllm_model,
                                                   vllm_sampling_params,
                                                   max_new_tokens,
                                                   model,
                                                   self.accel,
                                                   num_gens,
                                                   prompt_list,
                                                   image_list,
                                                   len(prompt_list))
        output_texts = [p + c for p, c in zip(prompt_list * num_gens, completion_texts)]

        # postprocessing for text and image
        _inputs = processor(text=prompt_list * num_gens,
                            images=image_list * num_gens,
                            padding=True,
                            return_tensors="pt").to(self.accel.device)
        prompt_length = _inputs.input_ids.shape[1]

        # postprocessing for text and image
        new_prompt_list, new_image_list = Utility.postprocess(inputs * num_gens, processor, qa_index_list * num_gens,
                                                              completion_texts)
        _new_inputs = processor(text=new_prompt_list,
                                images=new_image_list,
                                padding=True,
                                return_tensors="pt").to(self.accel.device)

        # prompt + answer
        # just in case for that no answer is given
        if prompt_length == _new_inputs.input_ids.shape[1]: prompt_length -= 1
        # [prompt_length mighe be errorneous with +1 or -1 differnece for some samples, but it's fine]
        completion_ids = _new_inputs.input_ids[:, prompt_length:]
        completion_mask = _new_inputs.attention_mask[:, prompt_length:]

        # compute reward
        rewards = self.compute_reward(model=vllm_model,
                                         sampling_params=vllm_sampling_params,
                                         temperature=temperature,
                                         processor=processor,
                                         output_texts=output_texts,
                                         answers=[i['conversations'][qa_ind]['answer'] for i, qa_ind in
                                                  zip(inputs, qa_index_list)] * num_gens,
                                         accel=self.accel)
        rewards = torch.tensor(rewards).float().to(self.accel.device)
        rewards = rewards.view(-1, num_gens, 2)
        avg_reward = rewards.mean(dim=(0, 1))
        sum_rewards = rewards.sum(dim=2)
        advantages = ((sum_rewards.view(-1) - sum_rewards.mean(dim=1).repeat_interleave(num_gens)) / (
                    sum_rewards.std(dim=1).repeat_interleave(num_gens) + 1e-4)).unsqueeze(1)

        self.accel.print('----------------Example Generation----------------')
        self.accel.print(output_texts[0])
        self.accel.print('')
        self.accel.print('')
        self.accel.print(f'Reward-Ans: {rewards[0][0][0]}, Reward-Format: {rewards[0][0][1]}')
        self.accel.print(f'Advantage: {advantages[0][0]}')
        self.accel.print(f'Completion mask shape: {completion_mask.shape}')
        self.accel.print(f'Completion shape: {len(completion_texts)}')
        self.accel.print(f'prompt_length: {prompt_length}')
        self.accel.print(f'_new_inputs.input_ids shape: {_new_inputs.input_ids.shape}')
        self.accel.print('--------------------------------------------------')

        # per token logps
        with torch.no_grad():
            old_per_token_logps = self.compute_log_probs(model, _new_inputs, completion_ids.shape[1])
            ref_per_token_logps = self.compute_log_probs(ref_model, _new_inputs, completion_ids.shape[1])

        # GPU STOP and Memory optimization
        self.memory_optimization()

        # GRPO iterations
        grpo_loss_list = []
        for _ in range(grpo_iters):
            # GRPO Loss per iteration
            new_per_token_logps = self.compute_log_probs(model, _new_inputs, completion_ids.shape[1])
            ratio = torch.exp(new_per_token_logps - old_per_token_logps)
            surrogate_loss = torch.min(ratio * advantages,
                                       torch.clamp(ratio, 1 - clip_low_eps, 1 + clip_high_eps) * advantages)
            kl = torch.exp(ref_per_token_logps - new_per_token_logps) - (ref_per_token_logps - new_per_token_logps) - 1
            per_token_loss = surrogate_loss - kld_beta * kl
            grpo_loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            # Backward
            self.accel.backward(grpo_loss)
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

            # listing to measure avg of grpo loss
            grpo_loss_list.append(grpo_loss.item())

            # GPU STOP and Memory optimization
            self.memory_optimization()

        # scheduler step
        scheduler.step()

        return {'GRPO-Loss': sum(grpo_loss_list) / len(grpo_loss_list), 'Reward-Ans': avg_reward[0].item(),
                'Reward-Format': avg_reward[1].item()}

    def compute_log_probs(self, model, _inputs, logits_to_keep, temp=0.9, chunk_size=64):
        # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        logits = model(**_inputs).logits[:, :-1, :]

        input_ids = _inputs.input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :] / temp

        """Process in chunks to reduce peak memory"""
        batch_size, seq_len, _ = logits.shape
        log_probs = torch.zeros(batch_size, seq_len, device=logits.device)

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_logits = logits[:, i:end_idx, :]
            chunk_ids = input_ids[:, i:end_idx]
            chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
            log_probs[:, i:end_idx] = chunk_log_probs.gather(
                dim=-1, index=chunk_ids.unsqueeze(-1)).squeeze(-1)
            del chunk_logits, chunk_log_probs
            torch.cuda.empty_cache()
        return log_probs

    # trial and error here, for better weightage between the reward components ...
    def compute_reward(self, model, sampling_params, temperature, processor, output_texts, answers, accel):
        # Rewards: 0 ~ 2
        functional_rewards = Utility.functional_reward_fn(model, sampling_params, temperature, processor,
                                                          output_texts, answers, accel)

        # Rewards: -1 ~ 2
        structural_rewards = Utility.structural_reward_fn(output_texts)

        combined_rewards = []
        for f_score, s_score in zip(functional_rewards, structural_rewards): combined_rewards.append(
            [f_score, s_score])
        return combined_rewards
