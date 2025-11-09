# TinyStories Language Model

A PyTorch implementation of a small language model trained on the TinyStories dataset, following the architecture described in the paper:

**"How Small Can Language Models Be and Still Speak Coherent English?"**  
by Ronen Eldan and Yuanzhi Li (2023)

[Paper Link](https://arxiv.org/abs/2305.07759) | [Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)

## Overview

This project implements a GPT-style transformer model with approximately 33 million parameters, designed to generate coherent English stories despite its small size. The model architecture follows the TinyStories-33M specifications from the paper, demonstrating that small language models can produce meaningful text when trained on appropriately designed datasets.

## Model Architecture

The model follows the TinyStories-33M architecture:

- **Layers**: 6 transformer blocks
- **Hidden Size**: 384
- **Attention Heads**: 6 per layer
- **Feed-forward Size**: 1,536 (4 × hidden size)
- **Context Length**: 512 tokens
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Activation Function**: GELU
- **Dropout Rate**: 0.1 (training), 0.0 (inference)
- **Weight Tying**: Shared embedding and output layer weights
- **Total Parameters**: ~33M

### Key Features

- **Weight Tying**: Token embedding and output projection share weights, reducing parameters
- **GPT-2 Style Initialization**: Proper weight initialization following GPT-2 methodology
- **Mixed Precision Training**: Automatic bfloat16/float16 support for faster training
- **Gradient Accumulation**: Supports large effective batch sizes with limited GPU memory

## Project Structure

```
TinyStories/
├── model/
│   ├── __init__.py
│   └── model.py          # Model architecture definition
├── train.py              # Training script
├── generate.py           # Text generation script
├── evaluate.py           # Model evaluation script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── results/             # Training outputs (created automatically)
    ├── checkpoints/     # Model checkpoints
    │   ├── best_model.pth
    │   └── checkpoint.pth
    └── logs/            # Training logs
        └── training_*.log
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch 2.0 or higher

### Setup

1. Clone or navigate to the project directory:
```bash
cd TinyStories
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "from model.model import Model, Config; print('Installation successful!')"
```

## Usage

### Training

Train the model on the TinyStories dataset:

```bash
python train.py
```

The training script will:
- Automatically download the TinyStories dataset from Hugging Face
- Split the data into train (95%) and validation (5%) sets
- Train the model with hyperparameters from the paper
- Save checkpoints and best model to `results/checkpoints/`
- Log training progress to `results/logs/`

#### Training Hyperparameters

The default training configuration follows the TinyStories paper:

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.1
- **Batch Size**: 32
- **Gradient Accumulation Steps**: 32 (effective batch size: 1,024)
- **Gradient Clipping**: 0.5
- **Warmup Steps**: 1,000
- **Learning Rate Schedule**: Linear warmup + cosine annealing
- **Mixed Precision**: bfloat16 (if supported) or float16

#### Customizing Training

You can modify the training parameters by editing `train.py` or modifying the function call at the bottom of the file:

```python
train(
    epochs=20,                    # Number of training epochs
    batch_size=32,                # Batch size
    learning_rate=1e-4,           # Learning rate
    weight_decay=0.1,            # Weight decay
    eval_interval=500,            # Evaluation frequency (iterations)
    eval_iters=200,               # Number of batches for evaluation
    gradient_accumulation_steps=32,  # Gradient accumulation
    max_grad_norm=0.5,           # Gradient clipping threshold
    warmup_steps=1000,            # Warmup steps
    save_interval=5000,           # Checkpoint save frequency
)
```

### Text Generation

Generate text using a trained model:

```bash
python generate.py --prompt "Once upon a time" --max_tokens 256
```

#### Generation Options

```bash
python generate.py \
    --prompt "Your story prompt here" \
    --model results/checkpoints/best_model.pth \
    --max_tokens 256 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95
```

**Parameters:**
- `--prompt`: Input text prompt (default: "Once upon a time")
- `--model`: Path to model checkpoint (default: `results/checkpoints/best_model.pth`)
- `--max_tokens`: Maximum tokens to generate (default: 256)
- `--temperature`: Sampling temperature, 0.0-2.0 (default: 0.8)
  - Lower = more deterministic, Higher = more creative
- `--top_k`: Top-k sampling, keep top k tokens (default: 50)
- `--top_p`: Nucleus sampling, cumulative probability threshold (default: 0.95)

### Evaluation

Evaluate model performance on validation set:

```bash
python evaluate.py --model results/checkpoints/best_model.pth
```

#### Evaluation Options

```bash
python evaluate.py \
    --model results/checkpoints/best_model.pth \
    --split validation \
    --batch_size 32 \
    --max_samples 1000 \
    --output results/evaluation.json
```

**Parameters:**
- `--model`: Path to model checkpoint
- `--split`: Dataset split to evaluate on (`train`, `validation`, or `all`)
- `--batch_size`: Batch size for evaluation
- `--max_samples`: Maximum number of examples to evaluate (None for all)
- `--output`: Path to save evaluation results JSON

## Results

### Where to Find Results

After training, results are saved in the `results/` directory:

- **Best Model**: `results/checkpoints/best_model.pth`
  - The model with the lowest validation loss
  - Use this for inference and evaluation

- **Training Checkpoint**: `results/checkpoints/checkpoint.pth`
  - Latest training state (model, optimizer, iteration, etc.)
  - Used for resuming training

- **Training Logs**: `results/logs/training_YYYYMMDD_HHMMSS.log`
  - JSON file with training history
  - Contains loss, validation metrics, learning rates per evaluation

### Expected Performance

Based on the TinyStories paper, you should expect:

- **Training Loss**: Should decrease steadily during training
- **Validation Loss**: Should track training loss closely (indicating good generalization)
- **Perplexity**: Should reach reasonable values (lower is better)
- **Generation Quality**: After sufficient training, the model should generate coherent short stories

## Training Tips

1. **GPU Memory**: If you encounter out-of-memory errors:
   - Reduce `batch_size`
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Reduce `block_size` in the config (though this may affect performance)

2. **Training Time**: Training on a single GPU typically takes:
   - Several hours to days depending on hardware
   - Monitor validation loss to determine when to stop

3. **Resuming Training**: The script automatically saves checkpoints. If training is interrupted:
   - Simply run `python train.py` again
   - It will automatically resume from the last checkpoint

4. **Monitoring**: Watch the training logs for:
   - Decreasing loss (both train and validation)
   - Validation loss should be close to training loss
   - Learning rate schedule progression

## Model Details

### Architecture Components

1. **Token Embedding**: Maps input tokens to dense vectors
2. **Positional Embedding**: Adds positional information
3. **Transformer Blocks**: Each contains:
   - Multi-head self-attention (6 heads)
   - Feed-forward network (4× expansion with GELU)
   - Layer normalization and residual connections
4. **Output Projection**: Uses weight-tying with token embeddings

### Implementation Notes

- The model uses PyTorch's `MultiheadAttention` for efficient attention computation
- Weight initialization follows GPT-2 style (normal distribution, std=0.02)
- Mixed precision training is automatically enabled when supported
- Model compilation (`torch.compile`) is used for faster training (PyTorch 2.0+)

## Citation

If you use this implementation, please cite the original TinyStories paper:

```bibtex
@article{eldan2023tinystories,
  title={How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

## License

This implementation is provided for educational and research purposes. Please refer to the original paper and dataset licenses for usage terms.

## Acknowledgments

- Original TinyStories paper by Ronen Eldan and Yuanzhi Li
- Hugging Face for the datasets library and tokenizers
- PyTorch team for the deep learning framework

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or increase gradient accumulation steps
   - Use CPU training (slower but works)

2. **Dataset Download Issues**
   - Ensure internet connection
   - Check Hugging Face dataset availability
   - Try downloading manually: `python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories')"`

3. **Model Not Generating Well**
   - Ensure sufficient training (check validation loss)
   - Try different temperature/top_k/top_p values
   - Verify model checkpoint was saved correctly

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the project repository.

