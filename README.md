# RuralVet-LLM: 한국 농촌 수의료 인공지능 어시스턴트

> **한국어 수의학 도메인 특화 LLM 파인튜닝 및 양자화 연구**  
> Phase 0 Zero-Shot 베이스라인 → Phase 1 LoRA/QLoRA × Rank 비교 → Phase 2 7가지 양자화 방식 통합 분석

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

---

## 📋 개요

RuralVet-LLM은 **한국 농촌 지역의 수의료 접근성 격차**를 해소하기 위한 인공지능 어시스턴트 개발 연구입니다.

### 핵심 성과
- ✅ **ROUGE-L 84.5% 향상**: Zero-shot (0.0897) → LoRA rank=16 (0.1655)
- ✅ **7가지 양자화 방식 통합 비교**: FP16, Float8, 8bit, 4bit(NF4), 2bit(HQQ)
- ✅ **농촌 배포 실증**: 4-bit BNB 양자화로 VRAM 5.4GB, 36.0 tok/s 달성
- ✅ **기술적 난관 직면 및 공개**: 평가 버그 수정, GGUF 변환 실패, LLM-Judge 거부 분석

### 기초 모델
- **Qwen/Qwen3.5-4B** (Qwen3.5 family, Hybrid Architecture: Gated DeltaNet + Gated Attention)
- 한국어 능력 + 경량 아키텍처 + 저비용 운영

### 데이터셋
- **2,476개 레코드** 구성 (8가지 유형)
  - block_detail (33.3%): 병리/역학/임상 증상
  - procedure_detail (16.8%): 진단 검사/처치 절차
  - overview, steps, clinical_scenario 등
- **동물 종별**: 돼지 47.1%, 개 15.8%, 소/반추동물 15%, 말 9.1%, 기타
- **난이도별**: 기초 20.6% → 중급 63.1% → 고급 16.2% (교육적 스캐폴딩)

---

## 🚀 빠른 시작

### 설치

```bash
# 저장소 복제
git clone https://github.com/YOUR_USERNAME/RuralVet-LLM.git
cd RuralVet-LLM

# 의존성 설치
pip install -r requirements.txt

# (선택) Google Colab 환경
# notebooks/ 폴더의 .ipynb 파일들을 Colab에 업로드해 실행 가능
```

### 실행 방법

#### 1️⃣ **Phase 0: Zero-Shot 베이스라인 측정**
```bash
jupyter notebook notebooks/00_phase0_phase1.ipynb
# 셀 1-3 실행 → baseline_results.json 생성
```

#### 2️⃣ **Phase 1: LoRA/QLoRA × Rank (4/8/16) 파인튜닝**
```bash
# 셀 4에서 EXPERIMENT 선택 (ex1_A ~ ex1_F)
EXPERIMENT = 'ex1_C'  # LoRA rank=16 (우승 조합)

jupyter notebook notebooks/00_phase0_phase1.ipynb
# 셀 4-9 순차 실행 → ex1_C/results.json, adapter/ 생성
```

#### 3️⃣ **Phase 2: 7가지 양자화 방식 비교**
```bash
jupyter notebook notebooks/02_phase2_quantization.ipynb
# 양자화 실험 실행 (FP16, 4bit NF4, 2bit HQQ 등)
# → phase2_results.json, visualizations 생성
```

### 결과 확인

```bash
# 최종 보고서 (상세 분석)
cat results/RuralVet_LLM_최종보고서.md

# 실험 결과
cat results/phase1_results.json      # Phase 1 비교
cat results/phase2_results.json      # Phase 2 양자화 비교
```

---

## 📊 주요 결과

### Phase 1: LoRA vs QLoRA × Rank

| 실험 | 방식 | Rank | ROUGE-L | BERTScore | 평가 |
|------|------|------|---------|-----------|------|
| ex1_C | **LoRA** | **16** | **0.1655** | **0.6737** | 🏆 우승 |
| ex1_B | LoRA | 8 | 0.1400 | 0.6651 | 2위 |
| ex1_E | QLoRA | 8 | 0.1264 | 0.6411 | 3위 |
| baseline | Zero-Shot | - | 0.0897 | 0.6503 | 🔴 기준 |

**핵심 발견:**
- LoRA > QLoRA (+1.44% ROUGE-L, 양자화 노이즈 비용)
- Rank 효과 명확 (r=4 → r=8 → r=16: +26.6% → +6.8% 수익 체감)
- 3 epoch, 2,171개 학습 데이터로 84.5% 향상

### Phase 2: 7가지 양자화 방식

| 포맷 | VRAM | tok/s | 품질 유지 | 용도 |
|------|------|-------|---------|------|
| 4bit NF4 | **5.4GB** | **36.0** | 95.7% | ✅ **농촌 배포 최적** |
| 8bit 표준 | 6.8GB | 39.5 | 98.2% | 조건부 (6GB 이상) |
| FP16 (베이스) | 10.5GB | 52.9 | 100% | 서버용 (부적합) |
| 2bit HQQ | 2.8GB | 18.2 | 76% | 극단적 압축 |

**배포 시나리오:**
- 🏘️ **농촌 엣지** (≤4GB): GGUF Q4_K_M (변환 실패 - 미검증)
- 🚜 **군 단위 센터** (6-8GB): 4bit BNB ✅ 실행 가능
- 🏛️ **서버** (≥12GB): FP16 베이스라인

---

## 🔬 연구 방법론

### 평가 지표
1. **ROUGE-L F1**: kiwipiepy 기반 한국어 형태소 분석 LCS 비교
2. **BERTScore**: klue/roberta-large 임베딩 의미 유사성
3. **배포 메트릭**: VRAM, 추론 속도(tok/s), 모델 크기

### 주요 기술적 결정

| 항목 | 선택 | 이유 |
|------|------|------|
| 기초 모델 | Qwen3.5-4B | 한국어 + 경량 + 하이브리드 아키텍처 |
| PEFT 방식 | LoRA | QLoRA 대비 11.4% 성능 우위, 학습 안정성 |
| Target 모듈 | q_proj, v_proj, k_proj, o_proj | Attention 레이어 (전체 ~75% 효과) |
| Rank 선택 | 16 | 수익 체감 고려 (r=8→16 +6.8% vs r=16→32 +2%) |
| 평가 템플릿 | ChatML (apply_chat_template) | **버그 수정**: Phase 2에서 미적용으로 인한 결과 무효화 |
| 생성 방식 | Greedy decoding | 재현성 확보 (sampling 제거) |

---

## ⚠️ 기술적 난관 & 학습

### 1. 평가 코드 버그 (가장 치명적)
**문제**: 채팅 템플릿 미적용 → FP16이 양자화 모델보다 낮은 ROUGE-L 표시  
**근본 원인**: LLM 평가에서 훈련 형식과 추론 형식 불일치  
**해결**: `apply_chat_template()` 적용, 결과 재실험, `FORCE_RERUN=True`  
**교훈**: 평가 파이프라인 검증을 훈련 전에 수행할 필요성

### 2. GGUF 변환 실패
**문제**: `missing tensor 'blk.32.attn_norm.weight'`  
**근본 원인**: Qwen3.5 하이브리드 아키텍처(MLA) ↔ llama.cpp 호환성 미성숙  
**현황**: 2026년 6월 기준 미해결 (llama.cpp 업데이트 대기)  
**영향**: CPU 기반 모바일 배포 경로 차단

### 3. LLM-Judge 실패
**문제**: Claude API가 수의학 콘텐츠 평가 거부 (안전 정책)  
**해결**: BERTScore 단독 사용 → 정성적 평가 백업 필요

### 4. Google Colab 환경 불안정
**문제**: OOM Killer, 드라이버 충돌, 12시간 런타임 제한  
**해결**: llama.cpp `use_mmap=False`, CUDA_VISIBLE_DEVICES 격리, 200스텝 체크포인트

---

## 📈 한계 & 향후 과제

### 현재 한계
- 📊 **낮은 절대 성능**: ROUGE-L 0.1655 (실용적 배포 기준 미달)
- 📂 **데이터 규모**: 2,171개 학습 데이터 (의료 도메인 기준 부족)
- 🔬 **임상 검증 부재**: 실제 수의사 평가 미수행
- 🚀 **배포 경로 제약**: GGUF 변환 실패 → 모바일 온디바이스 불가

### 향후 연구 로드맵
1. **RAG 결합**: 농촌진흥청/수의학 교재 벡터 DB 구축
2. **데이터 확대**: 5,000+ 전문가 검수 데이터셋
3. **도메인 루브릭**: 수의사 평가 (처방 정확성, 안전성, 실행 가능성)
4. **현장 파일럿**: 10명 수의사 2주 사용자 테스트
5. **모델 재선택**: Qwen3.5 대신 GGUF 호환 모델 (Llama-3.2-3B, Gemma-2-2B)

---

## 🏗️ 프로젝트 구조

```
RuralVet-LLM/
├── README.md (이 파일)
├── LICENSE (MIT)
├── CITATION.cff (학술 인용)
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 00_phase0_phase1.ipynb    # Phase 0 베이스라인 + Phase 1 LoRA/QLoRA
│   └── 02_phase2_quantization.ipynb  # Phase 2 양자화 7가지 방식 비교
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py (데이터셋 로드 유틸)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fine_tune.py (LoRA/QLoRA 파인튜닝)
│   │   └── quantization.py (양자화 함수)
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py (ROUGE-L, BERTScore 계산)
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── docs/
│   ├── METHODOLOGY.md (상세 방법론)
│   ├── ARCHITECTURE.md (Qwen3.5 하이브리드 아키텍처)
│   ├── DATASET.md (데이터셋 구성 상세)
│   ├── TROUBLESHOOTING.md (Colab 트러블슈팅)
│   └── RESULTS.md (실험 결과 해석)
│
├── results/
│   ├── RuralVet_LLM_최종보고서.md (224KB, 2,914줄 완전 분석 보고서)
│   ├── phase1_results.json
│   ├── phase2_results.json
│   └── visualizations/
│       ├── phase1_comparison.png
│       ├── phase2_pareto.png
│       └── ...
│
└── .github/
    └── workflows/
        └── tests.yml (CI/CD - 추후)
```

---

## 💻 시스템 요구사항

### 최소 사양 (평가만)
- Python 3.10+
- 8GB RAM
- 6GB VRAM (4bit 양자화 모델)

### 권장 사양 (파인튜닝)
- Python 3.10+
- 16GB RAM
- 16GB VRAM (V100/A100)
- Google Colab Pro (12시간 런타임)

### 의존성
```
transformers==5.9.0
peft==0.15.2
bitsandbytes==0.49.2
trl==0.16.1
torch>=2.0.0
datasets
rouge-score
bert-score
kiwipiepy
matplotlib
seaborn
pandas
```

---

## 📚 인용 (Citation)

학술 발표 또는 논문 작성 시 다음 형식으로 인용해주세요:

### BibTeX
```bibtex
@software{rh_ruravet_llm_2026,
  title = {RuralVet-LLM: Korean Rural Veterinary AI Assistant via LoRA/QLoRA Fine-tuning and Multi-method Quantization},
  author = {Roh, Hyeong-Woo},
  year = {2026},
  month = {6},
  url = {https://github.com/YOUR_USERNAME/RuralVet-LLM},
  doi = {10.5281/zenodo.XXXXXXX},  # (Zenodo 등록 후)
  note = {Phase 0-2 Complete Research Report}
}
```

### APA
```
Roh, H. W. (2026). RuralVet-LLM: Korean rural veterinary AI assistant via LoRA/QLoRA fine-tuning and multi-method quantization (Phase 0-2) [Computer software]. 
https://github.com/YOUR_USERNAME/RuralVet-LLM
```

---

## 📖 상세 문서

- **[연구 방법론](docs/METHODOLOGY.md)**: 3단계 실험 파이프라인, LoRA 이론, 양자화 기법
- **[Qwen3.5 아키텍처](docs/ARCHITECTURE.md)**: 하이브리드 아키텍처, Gated DeltaNet, MLA 주의 메커니즘
- **[데이터셋 분석](docs/DATASET.md)**: 8가지 유형, 난이도별/동물 종별 분포, 큐레이션 원칙
- **[실험 결과 해석](docs/RESULTS.md)**: Phase 1-2 결과, Pareto 최적, 배포 시나리오별 분석
- **[문제점 분석](docs/TROUBLESHOOTING.md)**: 평가 버그, GGUF 실패, LLM-Judge 거부, Colab 불안정성
- **[최종 보고서](results/RuralVet_LLM_최종보고서.md)**: 10장 종합 분석 (224KB)

---

## 🤝 기여 (Contributing)

이 프로젝트는 교육적·연구적 목적의 완성 보고서입니다.  
버그 리포트, 개선 제안, 논의는 [Issues](https://github.com/YOUR_USERNAME/RuralVet-LLM/issues)로 환영합니다.

자세한 기여 가이드는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

---

## 📜 라이센스

MIT License - [LICENSE](LICENSE) 참조

자유롭게 사용, 수정, 배포 가능하며, 저자 표시 및 라이센스 고지 필수입니다.

---

## 🙏 감사의 말

- **Qwen Team** (alibaba): 한국어 능력 우수한 Qwen3.5-4B 모델
- **HuggingFace**: transformers, peft, bitsandbytes 라이브러리
- **Google Colab**: 무료 GPU 실험 환경
- **농촌진흥청, 수의대학**: 데이터 배경 및 도메인 지식
- **10년 AI 양자화 연구 교수 관점**: 최종 보고서 작성

---

## 📧 연락처

- **작성자**: 노형우 (Hyeong-Woo Roh)
- **이메일**: uslm8591@gmail.com
- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

**마지막 업데이트**: 2026년 6월 8일  
**프로젝트 기간**: 2025년 12월 ~ 2026년 6월  
**상태**: ✅ Phase 0-2 완료, 정시 발표 및 보고서 작성 완료
