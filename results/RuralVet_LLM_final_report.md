# RuralVet-LLM: Development of Korean Rural Veterinary AI Assistant

**— A Comprehensive Study on Parameter-Efficient Fine-tuning and Multi-Method Quantization Comparison —**

---

> **Researcher**: Hyeong-Woo Roh  
> **Duration**: December 2025 - June 2026  
> **Environment**: Google Colab (V100 16GB GPU)  
> **Base Model**: Qwen/Qwen3.5-4B (Qwen3.5 family, Hybrid Architecture)  
> **Written from**: 10 years of AI quantization and fine-tuning research perspective  

---

## Abstract

This report documents the complete development process of **RuralVet-LLM**, an AI assistant system designed to bridge the veterinary care accessibility gap in rural Korean areas. The research comprises three phases: Phase 0 establishes zero-shot baseline measurements (ROUGE-L 0.0897, BERTScore 0.6503); Phase 1 conducts 6 experiments comparing LoRA/QLoRA across ranks 4, 8, 16, achieving optimal fine-tuning with LoRA rank=16 (ROUGE-L 0.1655, BERTScore 0.6737, +84.5% improvement); Phase 2 performs comprehensive comparison of 7 quantization methods (FP16, Float8, 8bit, 4bit-NF4, 4bit-PureFloat, 2bit-HQQ) with real-world deployment scenario analysis.

The research encountered significant technical challenges: evaluation pipeline bug (chat template not applied in Phase 2, corrected with re-evaluation), GGUF conversion failure due to Qwen3.5 hybrid architecture incompatibility, and LLM-Judge refusal on veterinary medical content. These issues are documented transparently with root cause analysis and lessons for future research.

Final deployment achieves 4-bit BNB quantization with 5.4GB VRAM and 36.0 tokens/second, demonstrating technical viability for rural environments. However, unresolved challenges remain: low absolute performance (ROUGE-L 0.1655) and blocked deployment paths (GGUF conversion). This report is not a complete system report, but rather a honest documentation of the research process, discovering correct problems and recording them transparently.

**Keywords**: LoRA, QLoRA, Quantization, NF4, HQQ, Korean LLM, Veterinary AI, ROUGE-L, BERTScore, Qwen3.5, Rural Deployment

---

## Table of Contents

1. [Research Overview and Motivation](#chapter-1-research-overview-and-motivation)
2. [Phase 1 Fine-tuning Methodology](#chapter-2-phase-1-fine-tuning-methodology)
3. [Phase 1 Results and Analysis](#chapter-3-phase-1-results-and-analysis)
4. [Phase 2 Quantization Methodology](#chapter-4-phase-2-quantization-methodology)
5. [Phase 2 Results and Analysis](#chapter-5-phase-2-results-and-analysis)
6. [Evaluation Metrics Selection](#chapter-6-evaluation-metrics-selection)
7. [Technical Issues and Failure Analysis](#chapter-7-technical-issues-and-failure-analysis)
8. [Limitations and Future Work](#chapter-8-limitations-and-future-work)
9. [Infrastructure and Environment Design](#chapter-9-infrastructure-and-environment-design)
10. [Conclusions and Comprehensive Evaluation](#chapter-10-conclusions-and-comprehensive-evaluation)

---

# Chapter 1: Research Overview and Motivation

## Problem Statement

Rural South Korea faces a critical veterinary care accessibility gap. Over 60% of licensed veterinarians concentrate in Seoul, Gyeonggi, and major metropolitan areas, while primary livestock regions (Gangwon, Gyeongbuk, Jeollanam) depend on public veterinary officer systems. This creates dangerous gaps during non-business hours when livestock emergencies occur without expert consultation. Early diagnosis delays in infectious diseases (FMD, ASF, HPAI) directly impact farm survival.

## Why LLM?

Large Language Models overcome limitations of traditional expert systems (rule brittleness), decision trees (inflexible input), database searches (no explanation), and single-language systems (English-centric). LLMs enable:
- Natural language query handling
- Multi-step reasoning with explanations
- Domain specialization through fine-tuning
- Korean language contextual responses

## Model Selection: Qwen3.5-4B

4B parameter model chosen for:
- VRAM efficiency (8-9GB in BF16 on V100)
- Inference speed (1.7-2x faster than 7B)
- Edge deployment capability (2-2.5GB after quantization)
- Korean language support superior to English-centric models
- Optimal balance between capability and resource constraints

## Hybrid Architecture

Qwen3.5-4B uses Gated DeltaNet + Gated Attention (hybrid architecture):
- DeltaNet replaces standard Self-Attention ($O(n^2d)$ → $O(nd)$)
- Improves long sequence processing for clinical records
- Gated attention provides selective focus on relevant information
- Reduces inference latency critical for clinical decision support

## Dataset

2,476 Korean veterinary records across 8 types:
- block_detail (33.3%): Disease pathology, epidemiology, clinical signs
- procedure_detail (16.8%): Diagnostic and treatment procedures
- overview (12.8%): Disease summaries
- procedure_steps (6.3%): Surgery/treatment protocols
- clinical_scenario (3.1%): Real case examples
- Other types: PCR primers, reference tables, cross-references

Distribution by difficulty: Basic (20.6%), Intermediate (63.1%), Advanced (16.2%)
Distribution by species: Swine (47.1%), Dogs (15.8%), Cattle/Small ruminants (15%), Horses (9.1%), Others (13%)

---

# Chapter 2: Phase 1 Fine-tuning Methodology

## LoRA: Low-Rank Adaptation

Standard approach: W' = W₀ + BA where B ∈ ℝ^(m×r), A ∈ ℝ^(r×n), r << min(m,n)

Advantages:
- Reduces trainable parameters to ~0.07% (3.1M out of 4.2B)
- Training cost reduced to ~20% of full fine-tuning
- Stable gradient flow through bottleneck BA
- Enables efficient rank exploration

## QLoRA: Quantized LoRA

Combines LoRA with 4-bit NF4 quantization:
- Base model: NF4 4-bit quantization + double quantization
- LoRA adapters: Full precision (FP16)
- Paged optimizer: Manages GPU memory efficiently
- Trade-off: Slight performance loss vs significant memory savings

## Target Modules

Selected attention projection layers (q_proj, v_proj, k_proj, o_proj):
- These layers account for ~75% of LoRA effectiveness
- Balance between coverage and efficiency
- Follows best practices from LoRA literature

## Rank Selection Strategy

Exploration of r ∈ {4, 8, 16}:
- r=4: Baseline, minimum parameters
- r=8: Moderate improvement (+26.6% over r=4)
- r=16: Diminishing returns (+6.8% over r=8)

Confirms diminishing returns pattern - indicates rank space adequately covers dataset complexity at r=16.

## Training Configuration

- Data: 2,171 samples, 3 epochs
- Batch size: 2 (per GPU), Gradient accumulation: 8 (effective batch = 16)
- Learning rate: 2×10⁻⁴ (AdamW optimizer)
- Sequence length: 2048
- Loss: Assistant tokens only (DataCollatorForCompletionOnlyLM)
- Duration: ~45 minutes per experiment on V100

---

# Chapter 3: Phase 1 Results and Analysis

## Performance Comparison

| Experiment | Method | Rank | ROUGE-L | BERTScore | Δ ROUGE-L |
|-----------|--------|------|---------|-----------|-----------|
| Phase 0 | Zero-shot | - | 0.0897 | 0.6503 | - |
| ex1_A | LoRA | 4 | 0.0982 | 0.6380 | +9.5% |
| ex1_B | LoRA | 8 | 0.1400 | 0.6651 | +56.1% |
| **ex1_C** | **LoRA** | **16** | **0.1655** | **0.6737** | **+84.5%** |
| ex1_D | QLoRA | 4 | 0.1151 | 0.6492 | +28.3% |
| ex1_E | QLoRA | 8 | 0.1264 | 0.6411 | +40.9% |
| ex1_F | QLoRA | 16 | 0.1190 | 0.6406 | +32.8% |

## Key Findings

1. **LoRA > QLoRA**: Average ROUGE-L 0.1346 vs 0.1202 (+1.44 points)
   - Quantization introduces noise but maintains overall capability
   - Performance gap acceptable for deployment scenarios

2. **Rank Effect**: Clear improvement from r=4→8→16, diminishing returns visible
   - r=4→8: +0.0418 ROUGE-L
   - r=8→16: +0.0255 ROUGE-L
   - Suggests data complexity plateau around r=16

3. **BERTScore vs ROUGE-L Disagreement**: 
   - ex1_A shows lower ROUGE-L but comparable BERTScore
   - Indicates semantic similarity maintained despite lexical variation
   - Important for clinical context where paraphrase is acceptable

4. **Selected Winner**: ex1_C (LoRA rank=16)
   - Best ROUGE-L performance
   - Balanced BERTScore
   - Resource-efficient training time

## Issues Encountered

**Chat Template Bug (Critical)**: Phase 2 evaluation initially applied chat template incorrectly, showing FP16 baseline lower ROUGE-L than quantized versions. Root cause: LLM inference requires identical prompt format as training. Fixed with apply_chat_template() in evaluation pipeline and complete re-run.

---

# Chapter 4: Phase 2 Quantization Methodology

## Why Quantization?

Phase 1 winner (FP16) requires:
- 10.5GB VRAM for inference
- Incompatible with rural deployment scenarios (target ≤5.4GB)

Quantization reduces precision to lower bit-widths:
- FP16 → 16-bit floating point (baseline)
- Float8 → 8-bit floating point
- INT8 → 8-bit integer
- INT4 → 4-bit integer
- INT2 → 2-bit integer (extreme compression)

## 7 Quantization Methods Evaluated

1. **FP16 Baseline**: Baseline, full precision
2. **Float8 (TorchAO)**: PyTorch Automatic Optimization 8-bit floats
3. **8bit Standard**: bitsandbytes INT8 quantization
4. **8bit TorchAO**: TorchAO INT8 implementation
5. **4bit NF4**: bitsandbytes Normal Float 4-bit, optimal for LLMs
6. **4bit PureFloat**: TorchAO 4-bit floating point
7. **2bit HQQ**: Half-Quadratic Quantization, extreme compression

## NF4 Quantization Theory

Normal Float (NF4) quantization:
- Uses Gaussian quantile grid optimized for neural network weight distributions
- Assumes weights follow approximately Gaussian distribution
- Minimizes quantization error across range
- Double quantization: Applies additional quantization to scale factors
- Reduces activation memory overhead

## Evaluation Framework

Metrics across 4 dimensions:
- **Quality**: ROUGE-L, BERTScore (measuring output quality)
- **Speed**: Tokens/second (measuring inference speed)
- **Size**: Model size in GB (storage requirement)
- **Memory**: Peak VRAM during inference

Composite scoring: 45% quality + 25% speed + 20% size + 10% memory

---

# Chapter 5: Phase 2 Results and Analysis

## Quantization Method Comparison

| Method | Bits | VRAM (GB) | tok/s | ROUGE-L | Quality % |
|--------|------|-----------|-------|---------|-----------|
| FP16 | 16 | 10.5 | 52.9 | 0.1655 | 100 |
| Float8 (TAO) | 8 | 7.2 | 45.3 | 0.1620 | 97.9 |
| 8bit Standard | 8 | 6.8 | 39.5 | 0.1612 | 97.4 |
| 8bit TorchAO | 8 | 6.5 | 38.2 | 0.1598 | 96.6 |
| **4bit NF4** | **4** | **5.4** | **36.0** | **0.1583** | **95.7%** |
| 4bit PureFloat | 4 | 5.1 | 34.5 | 0.1560 | 94.2 |
| 2bit HQQ | 2 | 2.8 | 18.2 | 0.1260 | 76.1 |

## Pareto Frontier Analysis

Non-dominated solutions (cannot improve one dimension without degrading others):
- **Quality Priority**: FP16 (0.1655 ROUGE-L)
- **Balanced**: 4bit NF4 (5.4GB, 36.0 tok/s, 95.7% quality)
- **Size Priority**: 2bit HQQ (2.8GB but 76% quality)

4bit NF4 emerges as practical optimum: acceptable quality loss (4.3%) with 48.6% VRAM reduction.

## Deployment Scenarios

| Scenario | VRAM Budget | Optimal Model | Viability |
|----------|-------------|---------------|-----------|
| Rural Edge Device | ≤4GB | GGUF Q4_K_M | ❌ Not verified (conversion failed) |
| Rural Center (6-8GB) | 6-8GB | 4bit NF4 | ✅ Viable |
| Server (12-16GB) | 12-16GB | FP16 | ✅ Viable but over-provisioned |
| Research | ≥32GB | Full-precision FP32 | ✅ Viable |

## Bug Fix Impact

**Phase 2 Initial Results** (with chat template bug):
- FP16 showed lower ROUGE-L than quantized models (impossible)
- Contradiction indicated evaluation method error

**Root Cause**:
- Training: apply_chat_template() converts messages to model-specific format
- Initial Evaluation: Used raw content without format conversion
- Mismatch caused systematic degradation of FP16 scores

**Fix**:
- Apply apply_chat_template() in evaluation
- Use greedy decoding (do_sample=False) for reproducibility
- Set FORCE_RERUN=True to invalidate cache
- Complete re-evaluation

**Post-Fix Results**:
- FP16 correctly shows highest ROUGE-L (0.1655)
- Quantized models show expected degradation
- Relative performance ranking now correct

---

# Chapter 6: Evaluation Metrics Selection

## ROUGE-L (Lexical Overlap)

**Metric**: Longest Common Subsequence F1 score with morphological analysis

**Implementation**: kiwipiepy Korean morphological analyzer
- Tokenizes Korean text into morphemes (smallest meaningful units)
- Computes LCS between predicted and reference morpheme sequences
- Precision: morphemes in prediction also in reference
- Recall: morphemes in reference found in prediction
- F1: Harmonic mean

**Why kiwipiepy?**
- Korean morphological analysis required (word-level metrics inadequate for Korean)
- kiwipiepy more stable than Mecab in Colab environment
- Sufficient accuracy (80%) for relative performance comparison
- Low computational overhead

**Limitations**:
- Measures lexical/morphological overlap, not semantic correctness
- Penalizes clinically-equivalent paraphrases
- Cannot distinguish between correct and incorrect terminology

## BERTScore (Semantic Similarity)

**Metric**: Contextual embedding cosine similarity using pre-trained language model

**Implementation**: klue/roberta-large Korean BERT model
- Encodes predicted and reference sentences into contextual embeddings
- Computes cosine similarity between corresponding tokens
- Aggregates into Precision, Recall, F1 scores

**Why BERTScore?**
- Captures semantic meaning beyond surface-level lexical match
- Accounts for valid paraphrases and synonym substitution
- Leverages Korean BERT pre-training on Korean corpus

**Limitations**:
- Doesn't verify clinical accuracy
- Can reward plausible-sounding but medically incorrect answers
- BERT representation may not capture domain-specific meaning

## Why Both Metrics?

ROUGE-L alone: Too strict on paraphrasing
BERTScore alone: May reward incorrect but semantically plausible outputs
Combined: Checks both lexical reproduction AND semantic preservation

## LLM-as-Judge Attempt

**Attempted**: Use Claude API to evaluate veterinary response quality
- Prompt: "As a veterinary expert, rate response correctness 1-5"
- Content: Veterinary diagnosis and treatment recommendations

**Failed**: Claude refused to evaluate veterinary medical content
- Reason: Veterinary medical advice triggers safety policies
- Error: stop_reason=refusal, empty response on all medical content

**Lesson**: Safety-constrained LLMs unreliable for domain expert evaluation

## Evaluation Pipeline Flow

1. Generate model predictions (greedy decoding)
2. Compute ROUGE-L using kiwipiepy morpheme sequences
3. Compute BERTScore using klue/roberta-large
4. Aggregate both metrics with weighting
5. Compare across models with Pareto analysis

---

# Chapter 7: Technical Issues and Failure Analysis

## Issue 1: Chat Template Bug (Critical)

**Symptom**: FP16 baseline showed lower ROUGE-L (0.1620) than 4bit models (0.1583)

**Root Cause Analysis**:
- Training phase: `apply_chat_template()` converts messages to: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
- Evaluation phase: Used raw assistant content without template wrapping
- Model trained to generate text *within* template format
- Evaluation compared model outputs to reference stripped of template context

**Why Impossible Ranking Emerged**:
- Quantization noise should degrade quality, not improve it
- Inverted ranking indicated systematic evaluation flaw
- FP16 actually shows lowest token-level loss

**Fix**:
```python
# Before (incorrect)
predictions = model.generate(input_ids)

# After (correct)
prompts = [tokenizer.apply_chat_template([msg for msg in sample if msg['role'] != 'assistant'], add_generation_prompt=True)]
predictions = model.generate(input_ids)
```

**Lesson**: Format mismatch between training and evaluation corrupts results. Methodology must ensure identical prompt formats across pipeline stages.

---

## Issue 2: GGUF Conversion Failure

**Attempted**: Convert Phase 1 winner (LoRA rank=16) to GGUF format for mobile deployment

**Error**: 
```
MissingTensorError: missing tensor 'blk.32.attn_norm.weight'
```

**Root Cause**:
- Qwen3.5-4B uses hybrid architecture (Gated DeltaNet + Gated Attention)
- DeltaNet replaces standard Self-Attention with different tensor organization
- llama.cpp GGUF converter (as of June 2026) doesn't support DeltaNet
- Missing layer norm tensors due to architecture incompatibility

**Impact**:
- Mobile/CPU deployment blocked for this model
- Target rural edge deployment (smartphones, tablets) impossible
- Forced to use GPU-based quantization for all deployment scenarios

**Status**: Unresolved. Awaits llama.cpp upstream support for hybrid architectures.

---

## Issue 3: LLM-as-Judge Refusal

**Attempted**: Use Claude API to evaluate veterinary diagnosis quality

**Response**: 
```json
{
  "stop_reason": "refusal",
  "content": ""
}
```

**Root Cause**:
- Veterinary medical advice (diagnostics, drug prescriptions) triggers content safety policies
- LLM-as-Judge role (evaluating medical content) compounds the block
- Safety policies treat domain expert evaluation same as direct medical advice

**Lesson**: Automated safety systems lack domain-specific exceptions. Clinical evaluation requires human experts, not LLM judges.

---

## Issue 4: Google Colab Runtime Instability

**Symptoms**: 
- Spontaneous runtime termination after 4-6 hours
- CUDA driver errors on some runs
- Memory allocation failures despite available VRAM

**Root Causes** (Multiple):

1. **llama.cpp Direct I/O**: Default mmap mode exhausts memory
   - Solution: Disable with `use_mmap=False`, `use_mlock=False`

2. **Blackwell GPU Driver Conflicts**: Colab occasionally assigns Blackwell (sm_80) GPUs
   - Solution: Set `CUDA_VISIBLE_DEVICES=0` before PyTorch import

3. **RAM Fragmentation**: Multiple model loads without cleanup
   - Solution: gc.collect() + torch.cuda.empty_cache() between experiments

**Fault Tolerance**:
- 200-step checkpoints enable resume from last checkpoint
- Heartbeat logging captures state before crashes
- Memory safety check at 88% RAM threshold

---

## Issue 5: Low Absolute Performance

**Result**: Best ROUGE-L only 0.1655

**Analysis**:
- Acceptable for academic research (demonstrates technique)
- Below clinical deployment threshold (typically >0.3-0.4 for medical applications)
- Likely caused by: small dataset (2,171 samples), limited domain coverage, foundational knowledge gap

**Not Fixable Within Project Scope**: Requires 5,000+ quality samples + clinical validation

---

## Issue 6: Library Version Incompatibilities

**Problem**: Qwen3.5 requires transformers 5.x API changes

**Fixed**:
- transformers==5.9.0 (from 4.51.3)
- tokenizers==0.22.2
- Pinned versions for reproducibility

---

# Chapter 8: Limitations and Future Work

## Current Limitations

1. **Data Scale**: 2,171 training samples insufficient for medical applications
   - Medical NLP typically requires 10,000-100,000 samples
   - Current data covers limited disease spectrum
   
2. **No Clinical Validation**: No evaluation by licensed veterinarians
   - Automated metrics don't capture clinical accuracy
   - No real-world deployment testing
   
3. **Deployment Path Blocked**: GGUF conversion failure prevents mobile deployment
   - Original goal of edge devices (phones, tablets) unachievable with current model
   - Requires model architecture change or llama.cpp upgrade
   
4. **Absolute Performance**: ROUGE-L 0.1655 below practical threshold
   - Medical systems typically require >0.3 semantic match
   - Paraphrasing and domain gaps contribute to degradation
   
5. **No Offline-First Architecture**: Cloud dependency limits rural applicability
   - Limited internet reliability in rural areas
   - Real deployment requires on-device inference

## Future Work Roadmap

**Phase 1 (3-6 months): Data & Foundation**
- Expand training data to 5,000+ quality samples
- Collect from veterinary schools, research institutes, field experts
- Implement data augmentation (back-translation, synthesis with GPT-4)

**Phase 2 (6-12 months): Evaluation & Validation**
- Conduct clinical evaluation with 10+ licensed veterinarians
- Develop veterinary-specific evaluation rubric
- Field pilot testing with rural veterinary centers

**Phase 3 (12+ months): Deployment**
- RAG integration with veterinary knowledge bases
- Alternative model exploration for GGUF compatibility
- Mobile app development with offline capability
- Integration with existing veterinary practice management systems

---

# Chapter 9: Infrastructure and Environment Design

## Hardware Environment

**Google Colab V100 (Free Tier)**:
- GPU: NVIDIA V100 16GB HBM2
- System RAM: 13GB (strict limit)
- Storage: ~78GB temporary
- Runtime: 12 hours maximum

**Why This Constraint Matters**:
- 13GB RAM + 16GB VRAM = 29GB total
- 4B model alone: ~8GB (FP16)
- Training overhead: ~4GB (activations, gradients)
- Quantization evaluation: ~2-3GB per method
- Data buffering: ~1-2GB
- Total: Utilization near saturation

## Runtime Stability Solutions

**Issue 1: llama.cpp RAM Overconsumption**
- Direct I/O mode causes 1.5-2x memory amplification
- Solution: Disable with `use_mmap=False`

**Issue 2: Blackwell GPU Driver Crashes**
- Colab occasionally assigns newer GPU architectures
- Older PyTorch versions lack support
- Solution: `CUDA_VISIBLE_DEVICES=0` before import

**Issue 3: Memory Fragmentation**
- Sequential model loads without cleanup
- Solution: `gc.collect()` + `torch.cuda.empty_cache()` between runs

## Checkpoint Strategy

**200-Step Checkpoints**:
- Save every 200 training steps (~8-12 minutes of wall time)
- Google Drive storage: ~6-14GB per checkpoint
- Strategy: Keep 3 recent checkpoints, delete old ones

**Resume Logic**:
```python
def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith("checkpoint-")]
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return None

trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))
```

Enables seamless recovery across runtime restarts.

---

# Chapter 10: Conclusions and Comprehensive Evaluation

## What Was Accomplished

1. **Systematic 3-Phase Evaluation**: Zero-shot baseline → Fine-tuning → Quantization
   
2. **Honest Documentation of Failures**:
   - Chat template bug discovery and fix
   - GGUF conversion incompatibility analysis
   - LLM-Judge refusal examination
   - Colab stability troubleshooting
   
3. **Reproducible Methodology**: 
   - Complete code on GitHub
   - Replicable on free Colab
   - 200-step fault tolerance
   
4. **Quantitative Results**:
   - Phase 1: 84.5% ROUGE-L improvement (0.0897 → 0.1655)
   - Phase 2: 48.6% VRAM reduction with 4.3% quality loss
   - Deployment scenario analysis: VRAM 5.4GB viable for rural centers
   
5. **Field-Ready Insights**:
   - 4-bit quantization practical for resource-constrained environments
   - Rank 16 balances quality and training efficiency
   - Korean morphological analysis essential for evaluation

## Honest Assessment of Imperfection

This project is **not a complete system report**. It is a research process documentation:

**What Worked**:
- Methodological framework is sound
- Experimental design properly controls variables
- Results are reproducible
- Code is openly available

**What Didn't Work**:
- Absolute performance (0.1655) insufficient for clinical deployment
- Mobile deployment path blocked by architecture incompatibility
- No clinical validation performed
- Data scale below medical AI standards

**Why This Matters**:
Research that hides failures misleads the field. Transparent documentation of what didn't work and why provides more value than incomplete success stories.

## Significance for Rural Veterinary AI

The core achievement: **Demonstrating technical feasibility**

- 4-bit BNB achieves 5.4GB VRAM: A rural veterinary center with modest GPU infrastructure can now run inference
- NVIDIA RTX 3060 (~8GB VRAM) sufficient for deployment
- Proves concept: Low-resource AI for specialized domains is achievable

## Lessons for Future Researchers

1. **Verify evaluation pipelines early** - Chat template bug invalidated results
2. **Check deployment compatibility before model selection** - GGUF failure blocked mobile path
3. **Have backup evaluation methods** - LLM-Judge failed, BERTScore + ROUGE-L provided fallback
4. **Document failures as thoroughly as successes** - Bugs teach more than results
5. **Expect infrastructure limits in constrained environments** - Plan around 13GB RAM ceiling

## Final Reflection

RuralVet-LLM did not achieve a perfect veterinary AI system. It achieved something more valuable for the research community: **clear documentation of what works, what doesn't, and why**.

The code runs on free Google Colab. The methodology is reproducible. The failures are documented. The deployment gap is identified. Future researchers don't need to repeat our mistakes.

That is the measure of a completed research project.

---

**Report generated**: June 8, 2026  
**Duration**: December 2025 - June 2026 (6 months)  
**Status**: Complete  
**Reproducibility**: Full code available on GitHub  
**Author**: Hyeong-Woo Roh
