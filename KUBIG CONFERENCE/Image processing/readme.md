# 청설(聽說): 내 이야기를 들어줘  
수어 ↔ 한국어 실시간 양방향 번역 시스템

👩‍💻 **팀원**  
김동욱, 김지엽, 김지원, 이예지, 이준언  
KUBIG(Korea University Data Science & AI Society), 고려대학교  
(ppuppu1200, ljykjy1309, jinnyk42, ye_ji, jel0206)@korea.ac.kr

## 🏁 프로젝트 소개

**주제 선정 배경**  
- 청각장애인들은 일상·긴급 상황에서 정보 접근성에 큰 불편을 겪고 있음
- 수어-자연어 간 실시간 자동 번역 기술의 필요성을 인식,  
  → ‘내 이야기를 들어줘’ 프로젝트 기획
  
**수어의 문법적 특성**  
- 의미 단위(gloss) 중심의 간결하고 직관적인 표현  
- 문법적 제약이 적고 자유로운 구조  
- 표정, 시선, 고개 등 비수지(손이 아닌) 표현이 핵심!

## 📂 데이터셋

- **Train Set:**  
  - 고유 문장 1,828개, 수어 영상 3,595개 (여러 signer·카메라 앵글)
- **Validation/Test Set:**  
  - 수어 영상 448개  
- **데이터 출처:**  
  - AI-Hub 재난안전 정보 수어 영상 데이터셋  
  - ‘화재 사고’ 관련 뉴스 문장 및 수어 영상  
- **형식:**  
  - 2D/3D keypoint 정보 포함 JSON 파일 (OpenPose 기반, 얼굴 keypoint 및 confidence score 제외)
  - 일부 3D keypoint, 비수지·형태소 정보 제공

## 🛠️ 파이프라인 & 모델 구조

**왜 Keypoint 기반인가?**  
- 영상 전체를 CNN 등으로 처리할 때 잡음·용량 이슈, 불필요한 정보 학습 문제 발생  
- 손, 팔 등 주요 부위의 움직임만으로 입력 차원 축소 & 학습 효율화!

**Preprocessing**  
- Face keypoint, confidence 제외  
- Z-score Normalization (Pose), Min-Max Normalization (Hand)  
- Skip Sampling으로 데이터 증강

**Sign2Text: Keypoint → 한국어**  
- BiGRU Encoder + Attention + GRU Decoder 기반 seq2seq  
- gloss 없이 자연어 직접 생성  
- 정규화, slot 마스킹, repetition penalty, label smoothing 등 적용  
- BLEU: 0.3254 / METEOR: 0.4900

**Text2Sign: 한국어 → Keypoint**  
- KoGPT2 기반 인코더, Multi-head 디코더 구조  
- 손 종류, gloss, 타이밍, 위치 등 분리 학습  
- 애니메이션 기반 수어 동작 생성  
- [Epoch 4] Val Loss: 0.0807, BLEU (pre-train): 0.2565

![image](./모델.png)


## 🚀 주요 결과

- Keypoint 기반 입력만으로도 효과적인 수어 번역 가능!
- 단방향이 아닌 쌍방향(양방향) 번역 시스템 설계  
- slot-based 전처리로 고유명사 등 noise 대응  
- 실제 시연 영상도 프로젝트에 포함

## 🌟 의의, 한계 & 향후 계획

**의의**  
- 경량화, 실시간성 확보  
- gloss 없이도 자연어 문장 생성 → 재난 알림 등 실제 활용 가능  
- Text2Sign 모델은 향후 3D 아바타, AR 등으로 확장 가능!

**한계**  
- FIRE(화재) 관련 문장만 포함 → 문맥 다양성 부족  
- 일부 고유명사 마스킹이 완벽하지 않아 번역 성능 한계  
- Text2Sign: 형태적 표현은 가능하지만 의미 정합성은 부족  
- 웹 기반 시연 시스템은 아직 미완성

**향후 계획**  
- 다양한 재난·일상 문장 포함 대규모 수어 데이터셋 확보  
- slot tagging·의미 기반 평가 지표 등 모델 고도화

## 📧 문의

팀 이메일: (ppuppu1200, ljykjy1309, jinnyk42, ye_ji, jel0206)@korea.ac.kr
