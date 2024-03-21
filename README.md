# GptImplementation - Building LLMs from Scratch
Basics of GPT model to create your own generating model

## Dependencies (assuming windows): 
`pip install pylzma numpy ipykernel jupyter torch --index-url {Find you r right version of pytorch based on cuda and cudnn installed(use nvidia-smi to get the version) here: https://pytorch.org/}`

If you don't have an NVIDIA GPU, then the `device` parameter will default to `'cpu'` since `device = 'cuda' if torch.cuda.is_available() else 'cpu'`. If device is defaulting to `'cpu'` that is fine, you will just experience slower runtimes.

## All the links you should need are in this repo. I will add detailed explanations as questions and issues are posted.

## Visual Studio 2022 (for lzma compression algo) - https://visualstudio.microsoft.com/downloads/

## OpenWebText Download - https://skylion007.github.io/OpenWebTextCorpus/

## Research Papers:
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf
A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf