# 기여 가이드 (Contributing Guide)

먼저 이 프로젝트에 관심을 가져주셔서 감사합니다! 🙏

RuralVet-LLM은 한국 농촌 수의료 AI 개발을 위한 **교육적·연구적 목적의 완성 보고서**입니다.
아래 가이드를 따라 기여해주세요.

---

## 📝 기여 가능 영역

### 1️⃣ **버그 리포트**
모델, 코드, 문서에서 발견한 오류를 보고해주세요.

**Issue 작성 예시:**
```markdown
**제목**: Phase 2 양자화 코드에서 NF4 float 타입 오류

**설명**: 
- 현상: notebooks/02_phase2_quantization.ipynb 셀 X 실행 시 TypeError 발생
- 환경: Python 3.10, transformers==5.9.0, bitsandbytes==0.49.2
- 에러 메시지: `[full error traceback]`
- 예상: 성공적으로 NF4 양자화 실행

**재현 단계**:
1. 노트북 열기
2. 셀 X 실행
3. 오류 발생
```

### 2️⃣ **개선 제안**
평가 지표, 실험 설계, 문서화 개선 아이디어를 제안해주세요.

**예시:**
- 다른 한국어 평가 지표 제안 (KoMETEOR, CIDEr 등)
- 추가 양자화 방식 실험 (GPTQ, AWQ)
- 데이터셋 확대 방안

### 3️⃣ **문서 개선**
오타 수정, 더 명확한 설명, 번역 개선

### 4️⃣ **코드 리팩토링**
가독성, 성능, 재사용성 개선 (단, 원본 결과 불변 유지)

---

## 🚀 기여 워크플로우

### Step 1: Fork & Clone
```bash
# GitHub에서 Fork
git clone https://github.com/YOUR_USERNAME/RuralVet-LLM.git
cd RuralVet-LLM
git remote add upstream https://github.com/ORIGINAL_OWNER/RuralVet-LLM.git
```

### Step 2: 브랜치 생성
```bash
git checkout -b feature/내-개선사항
# 또는
git checkout -b fix/버그-수정
```

**브랜치 네이밍 규칙:**
- `feature/`: 새로운 기능 (feature/new-quantization-method)
- `fix/`: 버그 수정 (fix/phase2-nf4-dtype-error)
- `docs/`: 문서 개선 (docs/korean-readme)
- `refactor/`: 코드 정리 (refactor/eval-metrics-module)

### Step 3: 변경 사항 커밋
```bash
git add .
git commit -m "feat: NF4 양자화 메모리 최적화 개선 (#42)"
```

**커밋 메시지 규칙:**
- 영문으로 작성 (또는 `feat(ko):` 로 한글 표시)
- 명확하고 간결하게
- 이슈 번호 참조 (자동 연결: `#42`)

**커밋 타입:**
- `feat:` 새 기능
- `fix:` 버그 수정
- `docs:` 문서
- `style:` 코드 스타일 (포맷팅)
- `refactor:` 코드 리팩토링
- `test:` 테스트 추가
- `perf:` 성능 개선

### Step 4: 최신 코드 동기화
```bash
git fetch upstream
git rebase upstream/main
```

### Step 5: Push & Pull Request
```bash
git push origin feature/내-개선사항
```

GitHub에서 Pull Request 작성:

**PR 템플릿:**
```markdown
## 🎯 목적
이 PR이 해결하는 문제나 개선사항 설명

## 📝 변경 사항
- 항목 1
- 항목 2
- 항목 3

## ✅ 검증 방법
1. 테스트 실행 방법
2. 예상 결과
3. 실제 결과 (스크린샷/로그)

## 📚 관련 이슈
Closes #42

## 🔍 체크리스트
- [ ] 코드 리뷰 완료
- [ ] 테스트 통과
- [ ] 문서 업데이트됨
- [ ] 새 의존성 없음 (또는 requirements.txt 수정됨)
```

---

## ✨ 코드 스타일 가이드

### Python 스타일
```python
# PEP 8 준수
# 라인 길이: 88자 (Black 기본값)

# 좋은 예
def calculate_rouge_l(predictions, references, batch_size=8):
    """ROUGE-L metric with kiwipiepy Korean analysis."""
    results = []
    for pred, ref in zip(predictions, references):
        score = compute_rouge(pred, ref)
        results.append(score)
    return results

# 나쁜 예
def rouge(p,r):  # 변수명 불명확
    return sum([x for x in p])/len(p)  # 한 줄에 모든 로직
```

### 포맷팅
```bash
# 자동 포맷팅
pip install black isort
black src/
isort src/

# Linting
pip install flake8
flake8 src/
```

### 주석 & 문서화
```python
def fine_tune_lora(model, dataset, rank=16):
    """
    LoRA fine-tuning with rank selection.
    
    Args:
        model (AutoModelForCausalLM): Base Qwen3.5-4B model
        dataset (Dataset): Training dataset (2,171 samples)
        rank (int): LoRA rank (default: 16)
    
    Returns:
        PeftModel: Fine-tuned model with LoRA adapter
    
    Example:
        >>> model = load_model('Qwen/Qwen3.5-4B')
        >>> ft_model = fine_tune_lora(model, train_dataset)
        >>> ft_model.save_pretrained('./adapter')
    """
    # 핵심 알고리즘 설명
    lora_config = LoraConfig(r=rank, lora_alpha=2*rank)
    return get_peft_model(model, lora_config)
```

---

## 🧪 테스트 작성

새 기능 추가 시 테스트 코드도 함께 작성해주세요:

```bash
# 테스트 실행
pytest tests/ -v

# 커버리지 확인
pytest tests/ --cov=src --cov-report=html
```

**테스트 파일 위치:** `tests/test_*.py`

**예시:**
```python
# tests/test_metrics.py
import pytest
from src.eval.metrics import calculate_rouge_l

def test_rouge_l_identical():
    """ROUGE-L should be 1.0 for identical strings."""
    pred = "농촌 수의사용 AI 어시스턴트"
    ref = "농촌 수의사용 AI 어시스턴트"
    score = calculate_rouge_l([pred], [ref])[0]
    assert score == 1.0

def test_rouge_l_empty():
    """ROUGE-L should handle empty strings gracefully."""
    score = calculate_rouge_l([""], [""])[0]
    assert 0 <= score <= 1
```

---

## 📊 Code Review Checklist

PR이 merge되기 전 다음을 확인합니다:

- ✅ **코드 품질**: 가독성, 중복 제거, 성능
- ✅ **테스트**: 새 코드에 대한 테스트 포함
- ✅ **문서화**: README, docstring 업데이트
- ✅ **실험 재현성**: 난수 시드 고정, 결과 확인
- ✅ **의존성**: requirements.txt 업데이트
- ✅ **라이선스**: MIT 호환 라이브러리만 추가

---

## 🤝 커뮤니티 가이드

### 존중하는 소통
- 모든 관점을 존중합니다
- 건설적인 비판을 환영합니다
- 톤이 중립적이고 친절합니다
- 개인 공격 없이 아이디어에 집중합니다

### 질문하기
```markdown
**좋은 질문 예시:**
- "Phase 1에서 왜 RSLoRA를 사용하셨나요?"
- "이 버그를 재현하는 최소한의 코드는?"

**피할 질문:**
- "왜 이렇게 복잡해?"
- "이거 안 되는데 누가 만들었어?"
```

---

## 📖 추가 리소스

- **[최종 보고서](results/RuralVet_LLM_최종보고서.md)**: 완전한 연구 배경
- **[방법론](docs/METHODOLOGY.md)**: LoRA/QLoRA/양자화 상세
- **[데이터셋 설명](docs/DATASET.md)**: 2,476개 레코드 구성
- **[문제점 분석](docs/TROUBLESHOOTING.md)**: 기술적 난관 & 학습

---

## 🏆 기여자 인정

모든 기여자는 다음에 기록됩니다:
- GitHub 자동 인정 (Contribution graph)
- README의 Contributors 섹션
- 릴리스 노트

```markdown
## Contributors

Thanks to these wonderful people for their contributions:

- [@contributor1](https://github.com/contributor1) - 버그 수정
- [@contributor2](https://github.com/contributor2) - 문서 개선
```

---

## ❓ 도움이 필요하신가요?

- **Issues**: 질문/제안 환영 (label: `question`, `discussion`)
- **Discussions**: 설계 논의 (GitHub Discussions)
- **Email**: uslm8591@gmail.com

---

**감사합니다! 이 프로젝트가 한국 농촌 수의료 AI 발전에 도움이 되길 바랍니다.** 🚀
