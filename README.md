# Multi-modal LLM fine-tuning

This is a good practice repo for people who want to have an experience in multi-modal LLM fine-tuning. We use the following
strategies to achieve the fine-tuning:

&emsp;✅ **Accelerate and DeepSpeed-ZeRO3**<br>
&emsp;✅ **GRPO (Group Relative Policy Optimization)**<br>
&emsp;✅ **PEFT: LoRA (Low Rank Approximation)**<br>
&emsp;✅ **vLLM inference engine**<br>

# :wrench: Installation
```bash
conda env create -f environment.yaml
conda activate mmllm-FT
pip uninstall torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
```

---

# Dataset and model weights
It is painful to find a proper training set and model, so for your convenience, the following model and dataset are what I found that can work properly.
It should be a good point to start.

Dataset: [Sujet-Finance-QA-Vision-100k](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-QA-Vision-100k/blob/main/README.md). \
VL-model: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

Both have been loaded automatically in [main.py](./main.py)
```python
parser.add_argument('--model_name', default="Qwen/Qwen2-VL-2B-Instruct", type=str)
processor = AutoProcessor.from_pretrained(args.model_name, padding_side='left', min_pixels=min_pixels, max_pixels=max_pixels)
```
and [dataloader.py](./dataloader.py)
```python
data = load_dataset("sujet-ai/Sujet-Finance-QA-Vision-100k")
```

---

# Enable thinking during fine-tuning
Reasoning / thinking can significantly improve the LLM performance, so I enable the
reasoning by the following code snippet.
```python
# System Prompt
messages = [
    {
        "role": "system",
        "content": self.string_connect("You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
                                          "Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags.",
                                          split='\n'),
    }
]

# question prompt
if 'image' in _input.keys():
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": _input['conversations'][qa_index]['question'].replace('<image>','')}, # <image> token remove
            ],
        },
    )
    image_list.append(_input['image'])
else:
    messages.append(
        {
            "role": "user",
            "content": _input['conversations'][qa_index]['question'],
        },
    )

# answer Prompt
messages.append(
    {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
    }
)

```
That's right, combining a simple < think > token with the [structure reward]() is the way to enable it.

### 💻 Training with multi-GPU 

```shell
# ds_accel.yaml is the config file for deepspeed zero3
bash train.sh
```
