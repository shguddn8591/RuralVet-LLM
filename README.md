# RuralVet-LLM

Korean veterinary AI assistant for rural areas using Qwen3.5-4B with LoRA fine-tuning and quantization.

## Overview

RuralVet-LLM is a research project on parameter-efficient fine-tuning and quantization of language models for the veterinary domain.

**Base Model**: Qwen/Qwen3.5-4B  
**Training Data**: 2,476 veterinary records in Korean  
**Evaluation Data**: 242 Q&A pairs  

## Results

| Phase | Method | ROUGE-L | BERTScore | Status |
|-------|--------|---------|-----------|--------|
| Phase 0 | Zero-shot | 0.0897 | 0.6503 | Baseline |
| Phase 1 | LoRA r=16 | 0.1655 | 0.6737 | Optimal |
| Phase 2 | 4bit NF4 | 0.1583 | 0.6480 | Deployment |

**Key Metrics (Phase 2)**:
- Model size: 5.4GB (4bit NF4)
- Inference speed: 36.0 tokens/second
- VRAM: 5.4GB (rural deployment viable)

## Quick Start

### Installation

```bash
git clone https://github.com/shguddn8591/RuralVet-LLM.git
cd RuralVet-LLM
pip install -r requirements.txt
```

### Run Notebooks

**Phase 0 & Phase 1** (Zero-shot + LoRA/QLoRA):
```bash
jupyter notebook notebooks/00_phase0_phase1.ipynb
```

**Phase 2** (Quantization comparison):
```bash
jupyter notebook notebooks/02_phase2_quantization.ipynb
```

Both notebooks run on Google Colab (V100 recommended).

## Experiment Design

**Phase 0**: Zero-shot baseline measurement
- Model: Qwen3.5-4B FP16
- Evaluation: 242 test samples

**Phase 1**: LoRA/QLoRA fine-tuning
- 6 experiments: LoRA/QLoRA × Rank (4, 8, 16)
- Training: 2,171 samples, 3 epochs
- Winner: LoRA rank=16 (+84.5% ROUGE-L improvement)

**Phase 2**: Quantization comparison
- 7 methods: FP16, Float8, 8bit, 4bit-NF4, 4bit-PureFloat, 2bit-HQQ
- Metrics: ROUGE-L, BERTScore, VRAM, inference speed
- Pareto analysis for deployment scenarios

## Evaluation Metrics

- ROUGE-L: Korean morphological analysis with kiwipiepy
- BERTScore: klue/roberta-large embeddings
- Deployment: VRAM usage and inference speed

## Technical Issues

**Fixed**:
- Chat template bug in Phase 2 evaluation (corrected with re-run)
- Greedy decoding for reproducibility

**Not Resolved**:
- GGUF conversion failure (Qwen3.5 architecture incompatibility)
- Low absolute performance (0.1655 ROUGE-L)
- No clinical validation

## Files

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
│   ├── RuralVet_LLM_최종보고서.md (224KB)
│   └── README.md
└── src/ (Python package structure)
```

## Requirements

Python 3.10+, PyTorch 2.0+

See `requirements.txt` for full dependency list.

## Citation

```bibtex
@software{rh_ruravet_llm_2026,
  title = {RuralVet-LLM: Korean Rural Veterinary AI with LoRA/QLoRA Fine-tuning and Quantization},
  author = {Roh, Hyeong-Woo},
  year = {2026},
  month = {6},
  url = {https://github.com/shguddn8591/RuralVet-LLM}
}
```

## License

MIT License - See LICENSE file for details.

## Limitations

- Training data limited to 2,171 samples
- No clinical expert validation
- GGUF conversion not supported (Qwen3.5 architecture issue)
- Model performance below practical deployment standards

## Future Work

1. Data scale-up to 5,000+ veterinary Q&A pairs
2. RAG integration with veterinary knowledge bases
3. Clinical validation with domain experts
4. Model switching for GGUF compatibility

## Contact

Author: 노형우 (Hyeong-Woo Roh)  
Email: uslm8591@gmail.com

---

**Complete research report**: See `results/RuralVet_LLM_최종보고서.md` (224KB, 10 chapters, full analysis)
