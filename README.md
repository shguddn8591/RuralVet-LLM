# RuralVet-LLM

Korean veterinary AI assistant for rural areas using Qwen3.5-4B with LoRA fine-tuning and quantization.

## Overview

RuralVet-LLM is a research project on parameter-efficient fine-tuning and quantization of language models for the veterinary domain.

**Base Model**: Qwen/Qwen3.5-4B  
**Training Data**: 2,476 veterinary records in Korean  
**Evaluation Data**: 242 Q&A pairs  
**Duration**: December 2025 - June 2026

## Results

| Phase | Method | ROUGE-L | BERTScore | Description |
|-------|--------|---------|-----------|-------------|
| Phase 0 | Zero-shot | 0.0897 | 0.6503 | Baseline |
| Phase 1 | LoRA rank=16 | 0.1655 | 0.6737 | Optimal (+84.5% improvement) |
| Phase 2 | 4bit NF4 | 0.1583 | 0.6480 | Deployment optimized |

**Deployment Metrics (Phase 2)**:
- Model size: 5.4 GB (4bit NF4 quantization)
- Inference speed: 36.0 tokens/second
- VRAM requirement: 5.4 GB (viable for rural scenarios)

## Quick Start

### Installation

```bash
git clone https://github.com/shguddn8591/RuralVet-LLM.git
cd RuralVet-LLM
pip install -r requirements.txt
```

### Run Experiments

**Phase 0 and Phase 1** (Zero-shot baseline + LoRA/QLoRA fine-tuning):
```bash
jupyter notebook notebooks/00_phase0_phase1.ipynb
```

**Phase 2** (Quantization comparison - 7 methods):
```bash
jupyter notebook notebooks/02_phase2_quantization.ipynb
```

Both notebooks are executable on Google Colab (V100 GPU recommended).

## Experimental Design

**Phase 0: Zero-shot Baseline**
- Model: Qwen3.5-4B in FP16
- Evaluation: 242 test samples
- Metrics: ROUGE-L and BERTScore

**Phase 1: Fine-tuning Methods**
- 6 experiments: LoRA and QLoRA with ranks 4, 8, 16
- Training data: 2,171 samples, 3 epochs
- Result: LoRA rank=16 achieved best performance (+84.5% ROUGE-L improvement)

**Phase 2: Quantization Comparison**
- 7 quantization methods: FP16, Float8, 8bit-Standard, 8bit-TorchAO, 4bit-NF4, 4bit-PureFloat, 2bit-HQQ
- Evaluation: ROUGE-L, BERTScore, VRAM usage, inference speed
- Analysis: Pareto frontier for deployment trade-offs

## Evaluation Metrics

- **ROUGE-L**: Longest Common Subsequence F1 with Korean morphological analysis (kiwipiepy)
- **BERTScore**: Semantic similarity using klue/roberta-large embeddings
- **Deployment Metrics**: VRAM consumption and tokens per second

## Technical Findings

**Successfully Resolved**:
- Chat template application bug in Phase 2 (corrected and re-evaluated)
- Reproducibility through greedy decoding strategy
- Checkpoint resume system for 12-hour runtime limits

**Known Issues**:
- GGUF conversion fails due to Qwen3.5 hybrid architecture incompatibility
- Absolute performance (0.1655 ROUGE-L) below practical deployment threshold
- No clinical validation with veterinary experts

## Project Structure

```
RuralVet-LLM/
├── README.md
├── LICENSE (MIT)
├── CITATION.cff
├── CONTRIBUTING.md
├── requirements.txt
├── notebooks/
│   ├── 00_phase0_phase1.ipynb
│   ├── 02_phase2_quantization.ipynb
│   └── README.md
├── docs/
│   └── README.md
├── results/
│   ├── RuralVet_LLM_final_report.md (224 KB)
│   └── README.md
└── src/
    ├── data/
    ├── models/
    ├── eval/
    └── utils/
```

## System Requirements

- Python 3.10+
- PyTorch 2.0+
- 6 GB VRAM (for 4bit quantized models)
- 13 GB system RAM (Google Colab environment)

See requirements.txt for complete dependency list.

## Citation

```bibtex
@software{roh_ruravet_llm_2026,
  title = {RuralVet-LLM: Korean Rural Veterinary AI with LoRA/QLoRA Fine-tuning and Multi-method Quantization},
  author = {Roh, Hyeong-Woo},
  year = {2026},
  month = {6},
  url = {https://github.com/shguddn8591/RuralVet-LLM}
}
```

## License

MIT License - See LICENSE file for details.

## Limitations

- Training data size (2,171 samples) insufficient for production deployment
- No clinical validation with domain experts
- GGUF conversion not supported (Qwen3.5 architecture compatibility issue)
- Absolute model performance below industry standards for clinical applications

## Future Research

1. Expand training data to 5,000+ veterinary Q&A pairs with expert review
2. Integrate Retrieval-Augmented Generation (RAG) with veterinary knowledge bases
3. Conduct clinical validation with licensed veterinarians
4. Explore compatible model architectures for GGUF conversion
5. Implement field deployment pilot with rural veterinary centers

## Contact

Author: Hyeong-Woo Roh  
Email: uslm8591@gmail.com  
GitHub: https://github.com/shguddn8591

---

**Complete Technical Report**: See results/RuralVet_LLM_final_report.md (224 KB, 10 chapters with comprehensive analysis)
