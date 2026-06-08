# Notebooks

This folder contains executable Jupyter notebooks for Phase 0-2 experiments.

## Files

### 00_phase0_phase1.ipynb

Phase 0 (zero-shot baseline) and Phase 1 (LoRA/QLoRA fine-tuning)

**Duration**: 2-3 hours

**Contents**:
- Phase 0: Zero-shot baseline measurement (3 cells, 10 minutes)
- Phase 1: LoRA/QLoRA fine-tuning with 6 experiments (15 cells, 2-3 hours)
- Analysis: Results comparison and visualization (2 cells, 5 minutes)

**How to Run**:
1. Open in Google Colab or Jupyter
2. Install dependencies (cell 0)
3. For Phase 1: Change `EXPERIMENT` variable in cell 4 (ex1_A through ex1_F)
4. Run cells sequentially

**Requirements**:
- Google Colab: V100 GPU recommended
- Local: 16GB VRAM minimum

### 02_phase2_quantization.ipynb

Phase 2: Quantization comparison (7 methods)

**Duration**: 4-6 hours

**Contents**:
- Environment setup and model loading
- 7 quantization methods: FP16, Float8, 8bit-Standard, 8bit-TorchAO, 4bit-NF4, 4bit-PureFloat, 2bit-HQQ
- Evaluation with ROUGE-L and BERTScore
- Results analysis and visualization

**How to Run**:
1. Open in Google Colab (strongly recommended due to runtime length)
2. Run cells sequentially
3. Set `FORCE_RERUN=True` to skip cached results

**Requirements**:
- Colab Pro (12-hour runtime) recommended
- V100 or A100 GPU

## Tips

- Save intermediate results to Google Drive
- Use 200-step checkpoints for fault tolerance
- Check memory usage with monitoring cells
- Results are cached; use FORCE_RERUN to re-evaluate

## Troubleshooting

**CUDA Out of Memory**:
- Reduce BATCH_SIZE in configuration cell
- Increase GRAD_ACCUM for same effective batch size

**Google Drive Mount Failed**:
- Use force_remount=True parameter
- Check storage quota

**Transformers Version Mismatch**:
- Install exact version: `pip install transformers==5.9.0`

## Output

Notebooks generate:
- JSON results files
- Visualization charts (PNG)
- Training logs and metrics
- Trained model weights or adapters
