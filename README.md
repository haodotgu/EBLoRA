# EBLoRA

This is the official implementation of our paper **["Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation"](https://icml.cc/virtual/2026/poster/61996)**.

Our paper was presented as an **ICML 2026 Poster**.

EBLoRA is a continual learning method for vision-language models (VLMs). This repository contains our LLaVA-1.5 based training and evaluation code, together with the scripts used in our experiments on the UCIT and MLLM-DCL benchmarks.

## Highlights

- Official code release for *Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation*
- Built on top of [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA)
- Supports continual training on two MLLM benchmarks:
  - [UCIT](https://github.com/Ghy0501/HiDe-LLaVA)
  - [MLLM-DCL](https://github.com/bjzhb666/MLLM-CL)
- Includes scripts for gradient-space extraction, continual fine-tuning, and sequential evaluation

## News

- Initial official repository release

## Environment Setup

We recommend creating a fresh Python environment with Python 3.10.

```bash
conda create -n eblora python=3.10 -y
conda activate eblora

pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
```

The training code depends on PyTorch, Transformers, PEFT, and DeepSpeed. Please install a CUDA-compatible PyTorch build for your machine before running large-scale training.

## Model Preparation

Our current release reproduces experiments based on **LLaVA-1.5-7B**.

```bash
cd /mnt/haogu/EBLoRA
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./models/llava-v1.5-7b
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ./models/clip-vit-large-patch14-336
```

After downloading the model, please update the LLaVA config under `./models/llava-v1.5-7b`:

1. Add the following fields to the model config:
   - `"mm_text_select_layer": -1`
   - `"mm_text_tower": "./models/clip-vit-large-patch14-336"`
   - `"mm_vision_tower": "./models/clip-vit-large-patch14-336"`
2. Remove the following fields from `generation_config.json` if they exist:
   - `"temperature": 0.9`
   - `"top_p": 0.6`

## Dataset Preparation

Please download the data from the original benchmark repositories:

- UCIT: [HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA)
- MLLM-DCL: [MLLM-CL](https://github.com/bjzhb666/MLLM-CL)

We expect the repository to be organized as follows:

```text
EBLoRA/
├── configs/
├── datasets/
│   ├── Domain_data/
│   │   ├── AD/
│   │   ├── Fin/
│   │   ├── Med/
│   │   ├── RS/
│   │   └── Sci/
│   └── UCIT/
│       ├── datasets/
│       │   ├── ArxivQA/
│       │   ├── CLEVR-Math/
│       │   ├── Flickr30k/
│       │   ├── IconQA/
│       │   ├── ImageNet-R/
│       │   └── VizWiz/
│       └── instructions/
│           ├── ArxivQA/
│           ├── CLEVR-Math/
│           ├── Flickr30k/
│           ├── IconQA/
│           ├── ImageNet-R/
│           └── VizWiz/
├── llava/
├── models/
│   ├── clip-vit-large-patch14-336/
│   └── llava-v1.5-7b/
├── results/
└── scripts/
```

In our server setup, `datasets/` and `models/` can also be symbolic links to shared storage, which is fully supported by the provided scripts.

## Repository Structure

- `llava/`: model, training, and evaluation code
- `configs/`: benchmark-specific training and evaluation configurations
- `scripts/Train_DCL/`: training pipeline for MLLM-DCL
- `scripts/Train_UCIT/`: training pipeline for UCIT
- `scripts/Eval_MLLM_DCL/`: evaluation scripts for MLLM-DCL
- `scripts/Eval_UCIT/`: evaluation scripts for UCIT
- `results/`: aggregated evaluation outputs
- `checkpoints/`: training checkpoints and extracted gradient subspaces

## Running Experiments

### MLLM-DCL

```bash
cd /mnt/haogu/EBLoRA
bash scripts/Train_DCL/train_all.sh
```

The default task order in `scripts/Train_DCL/train_all.sh` is:

1. RS
2. Med
3. AD
4. Sci
5. Fin

### UCIT

```bash
cd /mnt/haogu/EBLoRA
bash scripts/Train_UCIT/train_all.sh
```

The default task order in `scripts/Train_UCIT/train_all.sh` is:

1. ImageNet-R
2. ArxivQA
3. VizWiz
4. IconQA
5. CLEVR-Math
6. Flickr30k

## Notes on Compute Configuration

- The current training scripts launch distributed training with `torchrun --nproc_per_node=4`.
- If you want to use a different number of GPUs, please update both:
  - the `gpu_num` field in the corresponding JSON config under `configs/model_configs/`
  - the `--nproc_per_node` value in the corresponding task script under `scripts/Train_DCL/` or `scripts/Train_UCIT/`
- Extracted gradient subspaces are stored under checkpoint directories ending with `_gradients`.
- Final evaluation summaries are written to `./results`.

## Evaluation

The training pipelines already invoke benchmark-specific evaluation scripts after each task. You can also run them manually from `scripts/Eval_MLLM_DCL/` and `scripts/Eval_UCIT/` if you want to re-evaluate existing checkpoints.

## Acknowledgements

This repository is built upon the following excellent projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [HiDe-LLaVA / UCIT](https://github.com/Ghy0501/HiDe-LLaVA)
- [MLLM-CL / MLLM-DCL](https://github.com/bjzhb666/MLLM-CL)

We thank the original authors for making their code and datasets available.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{gu2026spectral,
  title={Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation},
  author={Gu, Hao and Luo, Mao-Lin and Zhou, Zi-Hao and Zhang, Han-Chen and Zhang, Min-Ling and Wei, Tong},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

