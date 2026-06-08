# 📓 Notebooks - Phase 0-2 실험 코드

이 폴더에는 RuralVet-LLM의 전체 실험 노트북이 포함되어 있습니다.
**Google Colab 또는 로컬 Jupyter에서 실행 가능**합니다.

---

## 📂 파일 구조

### 1. `00_phase0_phase1.ipynb` (통합 노트북)
**Phase 0 + Phase 1 실험** (약 2-3시간 소요)

**포함 내용:**
- **Phase 0** (셀 1-3): Zero-shot 베이스라인 측정
  - Qwen3.5-4B 모델 로드 (FP16)
  - 평가 데이터 242개 추론
  - ROUGE-L + BERTScore 계산
  - 결과: ROUGE-L 0.0897, BERTScore 0.6503

- **Phase 1** (셀 4-9): LoRA/QLoRA × Rank 파인튜닝
  - 6개 실험 (ex1_A ~ ex1_F)
  - LoRA: rank 4, 8, 16
  - QLoRA: rank 4, 8, 16
  - 각 실험 결과 저장
  - 최적 조합 선정: ex1_C (LoRA rank=16)

- **Phase 1 분석** (셀 10-11): 결과 비교 및 시각화
  - 6개 실험 비교 테이블
  - LoRA vs QLoRA 효과 분석
  - Rank별 성능 곡선
  - ROUGE-L vs BERTScore 산점도

**실행 방법:**

```python
# Colab
from google.colab import drive
drive.mount('/content/drive')

# 셀 4에서 실험 선택
EXPERIMENT = 'ex1_C'  # LoRA rank=16

# 셀 4-9 순차 실행
# → 각 실험 결과 Google Drive에 저장
```

---

### 2. `02_phase2_quantization.ipynb`
**Phase 2: 7가지 양자화 방식 비교** (약 4-6시간 소요)

**포함 내용:**
- 7가지 양자화 방식 정량화 및 평가
  - FP16 (baseline)
  - Float8 (TorchAO)
  - 8bit (standard bitsandbytes)
  - 8bit (TorchAO)
  - 4bit NF4 (bitsandbytes)
  - 4bit Pure Float
  - 2bit HQQ (Half-Quadratic Quantization)

- **평가 메트릭**
  - ROUGE-L (kiwipiepy 형태소 분석)
  - BERTScore (klue/roberta-large)
  - 배포 지표: VRAM, 추론 속도(tok/s), 모델 크기

- **분석**
  - 양자화 방식별 성능-효율 트레이드오프
  - Pareto 최적 분석
  - 배포 시나리오별 최적 모델 선정
  - 7가지 시각화 (차트, 표, 산점도)

**주요 결과:**
```
4bit NF4 (최적):
  - VRAM: 5.4GB (농촌 배포 가능 ✅)
  - 추론 속도: 36.0 tok/s
  - 품질 유지: 95.7% (vs FP16)
```

**실행 주의:**
⚠️ 이 노트북은 **장시간(4-6시간) 실행**되므로:
- Google Colab Pro 권장 (12시간 런타임)
- 중간 저장 및 체크포인트 기능 활용
- `FORCE_RERUN=True` 설정으로 기존 캐시 무효화

---

## 🚀 빠른 시작

### Colab에서 실행
```bash
# 1. Colab 열기
https://colab.research.google.com

# 2. GitHub에서 노트북 로드
File → Open Notebook → GitHub 탭
YOUR_USERNAME/RuralVet-LLM 검색

# 3. 셀 0부터 순차 실행
# (자동 설치 및 Drive 마운트)
```

### 로컬에서 실행
```bash
# 1. 저장소 복제
git clone https://github.com/YOUR_USERNAME/RuralVet-LLM.git
cd RuralVet-LLM

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Jupyter 실행
jupyter notebook notebooks/

# 4. 노트북 선택 및 실행
# 주의: 로컬에서 양자화 (VRAM 요구) 실험 시
# 최소 16GB GPU 메모리 필요
```

---

## 🔧 각 노트북의 주요 셀 구조

### Phase 0 & 1 노트북 (`00_phase0_phase1.ipynb`)

| 셀 | 내용 | 시간 | 출력 |
|----|------|------|------|
| 0 | 환경 설치 + Drive 마운트 | 30초 | ✅ |
| 1 | Phase 0 데이터 로드 | 5초 | eval.jsonl 242개 |
| 2 | Phase 0 모델 로드 + 추론 | 5분 | baseline_predictions |
| 3 | Phase 0 평가 (ROUGE-L, BERTScore) | 3분 | baseline_results.json |
| 4 | Phase 1 CONFIG (⭐ 여기서 실험 선택) | 1초 | `ex1_A` ~ `ex1_F` 선택 |
| 5 | Phase 1 데이터 + 모델 준비 | 2분 | LoRA/QLoRA 적용 완료 |
| 6 | Phase 1 SFT 학습 | 30-45분 | loss 그래프, 체크포인트 |
| 7 | Phase 1 FP16 머지 + 추론 | 5분 | 242개 추론 결과 |
| 8 | Phase 1 평가 계산 | 3분 | ROUGE-L, BERTScore |
| 9 | 결과 저장 | 1초 | results.json |
| 10 | 전체 결과 수집 | 5초 | 6개 실험 비교표 |
| 11 | 시각화 | 2분 | 4개 차트 생성 |

### Phase 2 노트북 (`02_phase2_quantization.ipynb`)

| 구간 | 내용 | 시간 |
|------|------|------|
| 셀 1-5 | 환경 + 데이터 + 기본 모델 로드 | 10분 |
| 셀 6-12 | 7가지 양자화 적용 및 평가 | 3-4시간 |
| 셀 13-15 | 결과 분석 및 시각화 | 10분 |

---

## 📊 체크포인트 & 재개

### Phase 1
각 실험마다 어댑터와 결과를 저장하므로:
- **중단 후 재개 가능**: 셀 4에서 실험 재선택 → 셀 4-9 재실행
- **모든 결과 Google Drive에 저장**: `RuralVet-LLM/results/phase1/{ex1_A,ex1_B,...}/`

### Phase 2
- **200스텝마다 체크포인트**: `checkpoints/checkpoint-200/`, `checkpoint-400/`
- **재개 함수**: `get_last_checkpoint()` 자동으로 마지막 지점에서 재개
- **FORCE_RERUN**: `True`로 설정 시 모든 캐시 무효화 및 재실행

---

## ⚠️ 주의사항

### Colab 환경
- **12시간 런타임 제한**: Phase 2는 분할 실행 권장
- **GPU 할당**: V100 안정적, A100 드라이버 호환성 주의
- **메모리**: 13GB RAM + 16GB VRAM 기준 설계

### 데이터셋
- `eval.jsonl`: 242개 (평가용)
- `train.jsonl`: 2,171개 (학습용)
- **포함 안 됨**: 실제 데이터셋은 Google Drive에만 저장

### 결과 신뢰성
- **Phase 2 버그 수정됨**: 채팅 템플릿 미적용 → 수정 완료
- **GGUF 변환 실패**: Qwen3.5 호환성 미성숙 (별도 해결 필요)
- **LLM-Judge 거부**: 수의학 콘텐츠 안전 정책 → BERTScore 단독 사용

---

## 📈 예상 결과

### Phase 0
```
ROUGE-L F1:   0.0897
BERTScore F1: 0.6503
```

### Phase 1 (최적: ex1_C)
```
Method:      LoRA
Rank:        16
ROUGE-L F1:  0.1655 (+84.5% vs baseline)
BERTScore:   0.6737 (+3.6% vs baseline)
```

### Phase 2 (최적: 4bit NF4)
```
VRAM:        5.4GB ✅ 농촌 배포 가능
tok/s:       36.0 (빠름)
품질:        95.7% (vs FP16)
```

---

## 🐛 트러블슈팅

### 문제: CUDA Out of Memory
```python
# 해결: 배치 크기 감소 (셀 4)
BATCH_SIZE = 1  # 2에서 1로 감소
GRAD_ACCUM = 16  # 누적 스텝 증가 (유효 배치 유지)
```

### 문제: 채팅 템플릿 오류
```python
# 원인: transformers 버전 불일치
# 해결: transformers==5.9.0 정확 설치
!pip install transformers==5.9.0 --upgrade
```

### 문제: Google Drive 마운트 실패
```python
# 해결: 강제 재마운트
drive.mount('/content/drive', force_remount=True)
```

더 자세한 문서: [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md)

---

## 📚 참고문헌

- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers & Pagnoni (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
- [최종 보고서](../results/RuralVet_LLM_최종보고서.md): 완전한 방법론 및 분석

---

**마지막 업데이트**: 2026년 6월 8일  
**상태**: ✅ Phase 0-2 완료, Colab에서 테스트됨
