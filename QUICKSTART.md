# Quick Start Guide

This guide will help you get started with training the TinyStories model quickly.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Installation

Test that everything is set up correctly:

```bash
python test.py
```

You should see:
- Model configuration details
- Parameter count (~33M)
- Successful forward pass
- Sample text generation

## Step 3: Start Training

Begin training the model:

```bash
python train.py
```

The script will:
1. Download the TinyStories dataset automatically
2. Preprocess and split the data
3. Start training with the default hyperparameters
4. Save checkpoints to `results/checkpoints/`
5. Log progress to `results/logs/`

### Training Output

You'll see output like:
```
Iter    500 | Train Loss: 3.2456 | Val Loss: 3.1892 | LR: 5.00e-05
  -> New best validation loss: 3.1892. Saving model...
```

## Step 4: Generate Text (After Training)

Once you have a trained model, generate text:

```bash
python generate.py --prompt "Once upon a time" --max_tokens 256
```

## Step 5: Evaluate Model

Evaluate your trained model:

```bash
python evaluate.py --model results/checkpoints/best_model.pth
```

## Results Location

All results are saved in the `results/` directory:

- **Best Model**: `results/checkpoints/best_model.pth`
  - Use this for inference
- **Checkpoint**: `results/checkpoints/checkpoint.pth`
  - For resuming training
- **Logs**: `results/logs/training_*.log`
  - Training history in JSON format

## Tips

1. **Monitor Training**: Watch the validation loss - it should decrease over time
2. **Stop Training**: Press Ctrl+C to stop. Training will resume from the last checkpoint when you run `train.py` again
3. **GPU Memory**: If you get OOM errors, reduce batch size in `train.py`
4. **Training Time**: Expect several hours to days depending on your hardware

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with different generation parameters in `generate.py`
- Modify hyperparameters in `train.py` to experiment with training

Happy training! 🚀

