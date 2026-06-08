# 📚 Documentation - 상세 기술 문서

이 폴더에는 RuralVet-LLM의 **상세한 기술 문서**들이 포함되어 있습니다.

---

## 📖 문서 목록

### 1. **[METHODOLOGY.md](METHODOLOGY.md)** - 연구 방법론 (필독)
RuralVet-LLM의 3단계 실험 파이프라인, 기술적 선택, 이론적 배경

**포함:**
- Phase 0-2 전체 파이프라인 설명
- LoRA vs QLoRA 이론 (W = W₀ + BA)
- Rank 선택 근거 및 수익 체감 분석
- 7가지 양자화 방식의 수학적 기초
- RSLoRA (Rank-Stabilized LoRA) 설명
- Target modules 선택 이유

### 2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Qwen3.5-4B 아키텍처
기초 모델의 구조와 특징, 왜 이 모델을 선택했는가

**포함:**
- Qwen3.5-4B 하이브리드 아키텍처
- Gated DeltaNet (효율적 시간 series 모델링)
- Gated Attention (선택적 주의)
- MLA (Multi-head Latent Attention)
- 한국어 능력 평가
- 다른 경량 모델과의 비교
- 모델 크기 & VRAM 요구사항

### 3. **[DATASET.md](DATASET.md)** - 데이터셋 상세 분석
2,476개 데이터의 구성, 큐레이션 원칙, 통계

**포함:**
- 8가지 데이터 유형 설명 (block_detail, procedure_detail 등)
- 난이도별 분포 (기초/중급/고급) 및 교육적 스캐폴딩
- 동물 종별 분포와 국내 축산 현실과의 정렬
- 데이터 품질 관리
- 예시 샘플 (5개)
- 향후 데이터 확대 방안

### 4. **[RESULTS.md](RESULTS.md)** - 실험 결과 해석
Phase 1-2 결과를 상세히 분석하고 해석

**포함:**
- Phase 1 6개 실험 상세 비교
- LoRA > QLoRA 이유 분석
- Rank 효과 곡선 및 수익 체감
- Phase 2 7가지 양자화 비교표
- Pareto 최적 분석
- 배포 시나리오별 최적 모델
- 성능-효율 트레이드오프 해석

### 5. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 문제점 분석 & 해결
기술적 난관, 버그, 환경 문제 및 해결책

**포함:**
- 평가 코드 버그 (채팅 템플릿 미적용)
- GGUF 변환 실패 (Qwen3.5 호환성)
- LLM-Judge 거부 (안전 정책)
- Google Colab 불안정성 (OOM, 드라이버)
- 각 문제의 근본 원인 분석
- 시도한 해결책과 결과
- 미해결 문제 및 향후 개선 방향

### 6. **[DEPLOYMENT.md](DEPLOYMENT.md)** (선택) - 배포 가이드
실제 배포 환경 설정 및 실행

**포함:**
- 엣지 디바이스 배포 (4GB ≤ VRAM ≤ 6GB)
- 서버 배포 (VRAM ≥ 12GB)
- Docker 컨테이너화
- API 서버 구축 (FastAPI)
- 성능 최적화 팁

---

## 🎯 문서 읽기 순서

### 처음 시작하는 경우
1. ← 이 README.md
2. 메인 [`../README.md`](../README.md) - 프로젝트 개요
3. [`METHODOLOGY.md`](METHODOLOGY.md) - 방법론 이해
4. 노트북 실행 [`../notebooks/README.md`](../notebooks/README.md)

### 깊이 있는 이해를 원하는 경우
1. [`ARCHITECTURE.md`](ARCHITECTURE.md) - 모델 선택 이유
2. [`DATASET.md`](DATASET.md) - 데이터 전략
3. [`METHODOLOGY.md`](METHODOLOGY.md) - 실험 설계
4. [`RESULTS.md`](RESULTS.md) - 결과 해석
5. [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - 기술적 난관 학습

### 배포를 계획하는 경우
1. [`RESULTS.md`](RESULTS.md) - 최적 모델 선정
2. [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - 알려진 문제
3. [`DEPLOYMENT.md`](DEPLOYMENT.md) - 배포 구성

### 관련 연구 진행 중
1. [`METHODOLOGY.md`](METHODOLOGY.md) - 기술 상세
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) - 모델 배경
3. [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - 피할 실수

---

## 📊 문서 크기 & 읽기 시간

| 문서 | 크기 | 읽기 시간 |
|------|------|----------|
| METHODOLOGY.md | ~15KB | 20분 |
| ARCHITECTURE.md | ~10KB | 15분 |
| DATASET.md | ~8KB | 10분 |
| RESULTS.md | ~12KB | 18분 |
| TROUBLESHOOTING.md | ~10KB | 15분 |
| DEPLOYMENT.md | ~8KB | 10분 |
| **합계** | **~63KB** | **88분** |

**최종 보고서** (결과 폴더): 224KB, 읽기 시간 3-4시간

---

## 🔍 주요 개념 찾기

### LoRA & QLoRA
→ [`METHODOLOGY.md` - 섹션 2.2-2.4](METHODOLOGY.md)

### Qwen3.5 하이브리드 아키텍처
→ [`ARCHITECTURE.md`](ARCHITECTURE.md)

### 데이터셋 구성
→ [`DATASET.md`](DATASET.md)

### 최적 모델 선택
→ [`RESULTS.md` - Pareto 분석](RESULTS.md)

### 배포 제약 조건
→ [`TROUBLESHOOTING.md` - GGUF 변환 실패](TROUBLESHOOTING.md)

---

## 💡 핵심 수치 한눈에

```
Phase 0 (Zero-Shot Baseline)
├─ ROUGE-L: 0.0897
└─ BERTScore: 0.6503

Phase 1 (LoRA/QLoRA × Rank)
├─ 최우수 (ex1_C): LoRA rank=16
├─ ROUGE-L: 0.1655 (+84.5%)
├─ BERTScore: 0.6737 (+3.6%)
├─ 학습 시간: ~45분/실험
└─ 학습 데이터: 2,171개

Phase 2 (7가지 양자화)
├─ 최적 모델: 4bit NF4
├─ VRAM: 5.4GB ✅
├─ 추론 속도: 36.0 tok/s
├─ 품질 유지: 95.7%
└─ 배포 가능성: 농촌 환경 조건부 적합

실험 환경
├─ GPU: V100 16GB
├─ RAM: 13GB (Colab)
├─ 런타임: 12시간 (Colab 무료)
└─ 비용: $0 (Google Colab 무료)
```

---

## 📝 문서 작성 규칙

이 문서들은 **교육적·학술적 명확성**을 중시합니다:

- ✅ **구체적 수치** 항상 포함 (0.1655 like this)
- ✅ **수식 설명** 마크다운으로 표현
- ✅ **코드 샘플** Python 문법 강조
- ✅ **표 사용** 비교 정리
- ✅ **참고문헌** 학술 출처 명시

---

## 🔗 외부 참고 자료

### 논문 & 기술 보고서
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models" [[PDF]](https://arxiv.org/abs/2106.09685)
- Dettmers & Pagnoni (2023): "QLoRA: Efficient Finetuning of Quantized LLMs" [[PDF]](https://arxiv.org/abs/2305.14314)
- Yao et al. (2024): "Qwen3 Technical Report" [[Link]](https://huggingface.co/Qwen)

### 도구 문서
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [bitsandbytes (Quantization)](https://github.com/TimDettmers/bitsandbytes)

### 평가 지표
- [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [kiwipiepy (Korean Morphological Analysis)](https://github.com/bab2min/kiwipiepy)

---

## ✉️ 질문 & 피드백

문서에 오류가 있거나 추가 설명이 필요하면:

1. **GitHub Issues**: 문제 리포트 (label: `docs`)
2. **Email**: uslm8591@gmail.com
3. **Discussions**: 설계/기술 논의

---

**마지막 업데이트**: 2026년 6월 8일  
**문서 상태**: ✅ 완성 및 검증됨
