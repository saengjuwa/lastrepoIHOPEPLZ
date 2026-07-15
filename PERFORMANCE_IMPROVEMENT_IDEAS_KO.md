[PERFORMANCE_IMPROVEMENT_IDEAS_KO.md](https://github.com/user-attachments/files/30041332/PERFORMANCE_IMPROVEMENT_IDEAS_KO.md)
# 랜드마크 유사도 성능 향상 기술 조사

이 문서는 현재 `ft_llm_edited` 코드를 기준으로, 한국 랜드마크를 직접 학습에 사용하지
않으면서 Triplet Accuracy를 높일 수 있는 방법을 우선순위대로 정리한 문서입니다.

논문에 성능 향상이 보고됐더라도 이 데이터에서 같은 폭으로 좋아진다고 보장할 수는
없습니다. 아래 방법은 반드시 **비한국 검증 데이터**로 한 가지씩 실험해야 합니다.

## 0. 현재 코드 구현 상태

이 문서가 처음 작성됐을 때의 기준선 설명은 비교 이유를 남기기 위해 그대로 둡니다.
현재 코드는 다음 후보를 실제 CLI 옵션으로 구현했습니다. 기본값에서 전부 동시에
활성화하지 않으며, 같은 고정 split에서 하나씩 비교해야 합니다.

- 공식 GLDv2 첫 N개 shard 다운로드, MD5, 안전한 선택 추출, 한국 label 제외, manifest/audit
- ArcFace와 Sub-center ArcFace
- singleton을 포함하는 classification 사전학습과 metric fine-tuning
- 300/384 등 가변 입력, 약한 색상·grayscale·회전·Random Erasing 증강
- DINOv2-S/B, patch token GeM, backbone 고정과 마지막 block 일부 해제
- log-domain Sinkhorn, dustbin, local/global descriptor를 쓰는 compact SALAD
- multi-dilation local attention, global GeM, orthogonal fusion을 쓰는 DOLG-style head
- Triplet, SupCon, 별도 학습 proxy를 쓰는 Proxy Anchor
- checkpoint에 queue까지 저장하는 Cross-Batch Memory
- epoch에 따라 hard-negative 사용 비율을 올리는 curriculum과 기존 FAISS 재탐색
- flip+five-crop 동시 TTA, multi-scale descriptor, 서로 다른 checkpoint score ensemble
- ALIKED/DISK/SuperPoint + LightGlue + homography inlier ratio의 선택적 지역 재정렬

명령은 [README.md](README.md)에 한 줄 PowerShell 형식으로 정리했습니다. 공식 SALAD
저장소의 GPL 코드는 복사하지 않고 논문의 Sinkhorn+dustbin 구조를 독립 구현했습니다.
DOLG는 원 논문의 모든 stage를 그대로 재현한 복제본이 아니라 핵심 지역/전역 결합을
구현한 DOLG-style 실험 head입니다. 따라서 발표 자료에도 이 이름을 그대로 써야 합니다.

다음 항목은 의도적으로 구현하지 않습니다.

- CosPlace: 현재 데이터에 위치·방향 metadata가 없습니다.
- 한국 채점 이미지 self-supervised 학습: 문제 규칙 위반 위험이 있습니다.
- 모든 후보의 동시 기본 활성화: 어느 기술이 효과를 냈는지 측정할 수 없습니다.

## 1. 현재 확인된 데이터 상태

현재 학습 CSV에는 8,259개 행과 7,497개 label이 있습니다. 이 중 6,900개 label은
사진이 한 장뿐이고, label당 두 장 이상인 것은 597개 label과 1,359개 행입니다.
사진이 네 장 이상인 label은 26개뿐입니다.

이 숫자는 현재 학습 컴퓨터에 있는 파일의 상태를 보여주는 참고 통계입니다. 이미지 원본은
다른 컴퓨터에서 다운로드·필터링한 뒤 이 컴퓨터로 옮기는 구조이므로, 이 통계만으로
외부 준비 데이터의 품질을 단정하지 않습니다.

## 2. 제약을 반영한 최종 우선순위

이 문서는 다음 제약을 받아들인 최종안입니다.

- 현재 받은 이미지와 CSV만 사용합니다.
- 이미지 다운로드와 한국 이미지 필터링은 다른 컴퓨터에서 수행합니다.
- 이 학습 컴퓨터에는 준비가 끝난 일부 이미지와 CSV를 가져옵니다.
- CSV에는 `country_code`가 없을 수 있습니다.

외부 컴퓨터에서 한국 이미지를 이미 제거했더라도, 이 프로젝트는 그 사실을 자동으로
알 수 없습니다. 현재 코드의 안전 검사는 `country_code` 열 또는 CSV의 `label`과 같은
번호를 적은 `korean_label_ids.txt`를 요구합니다. 외부 필터 로그와 CSV 해시를 함께
보관하고, 이 학습 컴퓨터에서 그 로그를 검사하는 전용 옵션을 추가하는 것이 가장
안전합니다. 검사를 삭제하거나 한국 채점 사진으로 확인하는 방법은 규칙 위반 위험이
있습니다.

또한 외부 컴퓨터에서 만든 CSV의 `path`는 이 컴퓨터의 `--image-root` 기준으로 열려야
합니다. 절대경로 대신 상대경로를 사용하면 컴퓨터가 바뀌어도 같은 모델 형식을 사용할
수 있습니다.


| 순위 | 기술 | 예상 효과 | 구현 난이도 | RTX 5070 비용 |
| ---: | --- | --- | --- | --- |
| 0 | 외부 PC에서 다운로드·한국 제외·CSV 생성 | 학습 가능 조건 | 낮음~중간 | 외부 PC |
| 1 | 현재 597개 label로 metric 학습 | 기준선 | 낮음 | 낮음 |
| 2 | Linear CE를 ArcFace로 변경 | 높음 | 중간 | 낮음 |
| 3 | singleton까지 쓰는 classification 사전학습 후 metric fine-tuning | 높음 가능 | 중간~높음 | 중간 |
| 4 | 입력 384와 적당한 증강 실험 | 중간 | 낮음 | 중간 |
| 5 | DINOv2-S/B + GeM 비교 | 높음 가능 | 중간 | 중간~높음 |
| 6 | Sub-center ArcFace 또는 Proxy Anchor | 중간~높음 | 중간 | 중간 |
| 7 | Cross-Batch Memory와 주기적 hard mining | 중간 | 중간~높음 | 중간 |
| 8 | DOLG 또는 SALAD 지역+전역 특징 | 높음 가능 | 높음 | 높음 |
| 9 | LightGlue 지역 특징 점수 결합 | 중간~높음 | 높음 | 추론 증가 |
| 10 | 다양한 모델 ensemble과 multi-scale TTA | 중간 | 낮음~중간 | 추론 증가 |

`예상 효과`는 현재 프로젝트에 대한 보장이 아니라, 현재 병목과 관련 논문 결과를
바탕으로 정한 실험 우선순위입니다.

## 3. 참고: 외부 데이터 준비 컴퓨터에서 가능한 개선

다음 내용은 현재 학습 컴퓨터에서 수행할 작업이 아닙니다. 이미지 다운로드 컴퓨터에서
이미지 선택 정책을 바꿀 수 있을 때의 참고사항입니다. 현재 사용자는 이 단계를 다시
수행하지 않는 것으로 가정하고, 아래의 모델·추론 실험으로 넘어갑니다.

현재 방식은 tar 일부를 받아 이미지 수는 확보했지만, 대부분의 landmark가 한 장씩만
남았습니다. Triplet 학습에서는 서로 다른 label 수만큼이나 **같은 label의 여러 사진**이
중요합니다.

같은 10GB 안에서 다음 목표로 다시 구성하는 것을 권장합니다.

```text
목표: label당 최소 4장, 가능하면 6~10장
예: 1,000개 label × 평균 8장 = 약 8,000장
```

권장 선택 절차는 다음과 같습니다.

1. 한국이 아닌 것으로 확인된 landmark ID만 남깁니다.
2. 해당 ID마다 다운로드 가능한 이미지가 최소 4장인지 확인합니다.
3. label마다 비슷한 장수를 받습니다.
4. 같은 URL, 같은 이미지 ID, 손상 이미지를 제거합니다.
5. 한 label 안에서도 시점, 거리, 계절이 다양한 사진을 우선합니다.
6. 마지막에 실제 파일이 존재하는 행만 CSV에 기록합니다.

GLDv2 논문은 정제된 `GLDv2-train-clean`으로 학습했을 때 다른 랜드마크 검색
데이터셋에서 GLDv1 학습보다 mAP가 최대 약 5% 개선됐다고 보고했습니다. 데이터 양만
늘리는 것보다 라벨 품질과 중복 제거가 중요하다는 근거입니다.

근거: [Google Landmarks Dataset v2, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Weyand_Google_Landmarks_Dataset_v2_-_A_Large-Scale_Benchmark_for_Instance-Level_CVPR_2020_paper.pdf)

## 4. 2순위: ArcFace 분류기로 교체

현재 모델은 일반 `Linear classifier + Cross Entropy`를 사용합니다. ArcFace는
정규화된 임베딩과 정규화된 class 중심 사이의 각도에 margin을 추가합니다.

쉽게 말하면 다음과 같습니다.

- 일반 CE: 정답 번호만 맞으면 됨
- ArcFace: 정답 번호를 맞히면서 다른 landmark와 각도 차이도 넉넉하게 벌려야 함

최종 평가는 cosine similarity를 사용하므로, 각도 공간을 직접 정리하는 ArcFace는 현재
평가 방식과 잘 맞습니다. GLDv2 공식 논문에서도 `ResNet101 + GeM + ArcFace`를
retrieval 기준선으로 사용했습니다.

현재 코드에서는 다음 변경이 필요합니다.

1. 일반 `nn.Linear` classifier를 cosine classifier로 변경합니다.
2. 정답 class 각도에 margin을 적용합니다.
3. ArcFace scale `s`와 margin `m`을 옵션으로 만듭니다.
4. 기존 Triplet Loss는 처음에는 그대로 함께 둡니다.

첫 실험 범위는 다음 정도가 안전합니다.

```text
s: 30 또는 64
m: 0.2, 0.3, 0.4
```

근거: [ArcFace, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)

## 5. noisy label 대응: Sub-center ArcFace

웹에서 모은 GLDv2에는 잘못된 label이나, 같은 label 안에서 전혀 다른 모습의 이미지가
포함될 수 있습니다. 일반 ArcFace는 class당 중심을 하나만 둡니다. Sub-center ArcFace는
class당 여러 중심을 두고 이미지가 가장 가까운 중심을 선택하게 합니다.

쉽게 말하면 한 랜드마크의 대표 얼굴을 하나만 두지 않고 다음처럼 여러 개 두는 것입니다.

- 정면 모습 중심
- 옆면 모습 중심
- 멀리서 본 모습 중심
- 잘못 붙은 이미지가 모이는 보조 중심

데이터를 완전히 정제하기 어렵다면 ArcFace 다음 실험으로 적합합니다. 단, class 수가
많을수록 classifier 메모리가 증가합니다. 현재처럼 실제 사용 label이 수백~수천 개라면
먼저 sub-center 2~3개로 실험할 수 있습니다.

근거: [Sub-center ArcFace, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)

## 6. Backbone을 DINOv2로 비교

현재 EfficientNetV2-S는 좋은 CNN 기준선입니다. 하지만 평가 대상은 학습에서 보지 못한
한국 랜드마크이므로, 새로운 환경으로 일반화하는 능력이 중요합니다.

DINOv2는 라벨 없이 대규모 자연 이미지에서 학습한 범용 시각 특징 모델입니다. AnyLoc은
DINOv2 patch 특징과 VLAD/GeM 같은 집계를 결합해 다양한 장소와 시점에서 강한 place
recognition 성능을 보고했습니다. SALAD도 DINOv2를 backbone으로 사용합니다.

권장 비교 순서는 다음과 같습니다.

1. DINOv2-S를 고정하고 GeM/projection만 학습
2. 마지막 transformer block 일부만 작은 LR로 해제
3. 메모리가 허용되면 DINOv2-B 비교
4. EfficientNetV2-S와 DINOv2의 score ensemble

5070 한 장에서는 큰 모델 전체 fine-tuning보다 작은 모델이나 부분 fine-tuning부터
시작하는 편이 안전합니다.

근거:

- [DINOv2](https://arxiv.org/abs/2304.07193)
- [AnyLoc](https://arxiv.org/abs/2308.00688)

외부 사전학습 모델은 문제 설명상 허용되지만, 모델명·가중치 출처·사전학습 데이터
설명을 PPT에 명시해야 합니다. 한국 채점 이미지로 self-supervised fine-tuning이나
test-time adaptation을 하는 것은 직접 학습으로 해석될 수 있으므로 사용하지 않는 편이
안전합니다.

## 7. 지역 특징과 전역 특징을 함께 사용

현재 GeM 임베딩은 사진 전체를 하나의 벡터로 요약합니다. 배경이 많거나 랜드마크가
작게 나온 사진에서는 중요한 창문, 첨탑, 문양 같은 지역 특징이 약해질 수 있습니다.

### DOLG

DOLG는 전역 특징과 attention으로 찾은 지역 특징을 서로 보완하도록 합친 뒤 하나의
compact descriptor를 만듭니다. 모델 하나로 cosine similarity를 계산할 수 있어 현재
파이프라인과 비교적 잘 맞습니다.

근거: [DOLG, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_DOLG_Single-Stage_Image_Retrieval_With_Deep_Orthogonal_Fusion_of_Local_ICCV_2021_paper.html)

### SALAD

SALAD는 DINOv2의 지역 patch들을 optimal transport 방식으로 cluster에 배정하고,
도움이 적은 배경 특징은 dustbin cluster로 버립니다. VPR 데이터셋에서 강한 single-stage
성능을 보고했지만, 구현과 메모리 비용은 GeM보다 큽니다.

근거: [SALAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.html)

추천 순서는 `DINOv2 + GeM`을 먼저 확인하고, 그 결과가 좋을 때 SALAD head를 붙이는
것입니다. Backbone과 집계 방식을 동시에 바꾸면 무엇이 효과가 있었는지 알기 어렵습니다.

## 8. Loss와 Negative 탐색 개선

### Cross-Batch Memory

현재 Batch-hard Triplet Loss는 batch 64장 안에서만 어려운 Negative를 찾습니다.
Cross-Batch Memory(XBM)는 이전 batch 임베딩을 작은 메모리 큐에 저장해, 현재 batch
밖에서도 어려운 Negative를 찾습니다.

일부 데이터 환경에서는 batch를 크게 만들기 어렵기 때문에 특히 유용한 후보입니다.
다만 오래된 임베딩과 현재 임베딩의 차이가 너무 커지지 않도록 queue 크기와 warm-up을
검증해야 합니다.

근거: [Cross-Batch Memory, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)

### Supervised Contrastive Loss

SupCon은 한 Anchor에 대해 같은 label의 여러 이미지를 모두 Positive로, 다른 label을
Negative로 사용합니다. Triplet 한 쌍만 고르는 것보다 batch의 관계를 더 많이 활용할 수
있습니다. 다만 같은 label 이미지가 충분해야 하므로 현재 데이터 재구성이 먼저입니다.

근거: [Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

### Proxy Anchor Loss

class별 학습 가능한 proxy를 두고 이미지와 proxy 관계를 학습합니다. 논문은 빠른 수렴과
noisy label/outlier에 대한 강건성을 보고했습니다. ArcFace/Sub-center ArcFace와 같은
실험에서 한 번에 모두 섞지 말고 대체 loss로 비교하는 것이 좋습니다.

근거: [Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)

### 현재 FAISS hard negative 개선

현재 hard-negative CSV는 한 번 생성한 뒤 fine-tuning에 사용합니다. 모델이 바뀌면
어려운 Negative도 바뀌므로 다음 curriculum을 실험할 수 있습니다.

1. baseline 학습
2. hard-negative 재탐색
3. 1~2 epoch fine-tuning
4. 새 checkpoint로 hard-negative 다시 탐색
5. 더 작은 LR로 마무리

처음부터 가장 어려운 Negative만 쓰면 label 오류를 강하게 학습할 수 있습니다. 초반에는
semi-hard Negative 비율을 높이고 후반에 hard 비율을 올리는 방법이 더 안전합니다.

## 9. 해상도와 데이터 증강

현재 입력은 300×300입니다. 랜드마크는 창문, 글자, 조각처럼 작은 구조가 중요하므로
384×384 또는 448×448가 도움이 될 수 있습니다.

권장 실험:

```text
E1: 300×300 baseline
E2: 같은 설정에서 384×384
E3: 300으로 학습 후 마지막 1~2 epoch만 384로 fine-tuning
```

해상도를 올리면 GPU 메모리와 시간이 크게 증가합니다. batch를 줄이면 한 batch 안의
Negative 수도 줄어들므로, XBM을 함께 쓰거나 `labels_per_batch`를 지나치게 낮추지 않아야
합니다.

현재 crop과 flip에 추가할 수 있는 약한 증강은 다음과 같습니다.

- 밝기·대비·채도 변화
- 약한 grayscale
- 약한 perspective 또는 rotation
- 작은 Random Erasing
- 낮은 강도의 RandAugment

세로 뒤집기, 강한 crop, 과도한 perspective, MixUp/CutMix은 건물 구조를 망가뜨리거나
두 landmark를 섞을 수 있으므로 후순위입니다.

근거: [RandAugment, CVPR Workshop 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)

## 10. 추론 성능 향상

### 현재 TTA 비교

다음 네 설정을 같은 checkpoint에서 바로 비교할 수 있습니다.

```text
none → flip → five_crop → five_crop_flip
```

`five_crop_flip`은 10개 view를 사용하므로 가장 느립니다. 랜드마크가 중앙에 작게 있는
사진에서는 모서리 crop이 오히려 점수를 낮출 수 있어 항상 최고라고 볼 수 없습니다.
비한국 고정 검증셋에서 정확도와 시간을 함께 기록해야 합니다.

### Multi-scale descriptor

같은 이미지를 예를 들어 300과 384 두 크기로 추론하고 임베딩을 평균낸 뒤 다시
정규화합니다. crop 위치를 늘리는 TTA와 다른 종류의 정보를 얻을 수 있습니다.

### 서로 다른 모델 ensemble

같은 구조의 비슷한 seed 모델만 합치는 것보다 다음처럼 성격이 다른 모델을 합치는 것이
보통 더 유용합니다.

- EfficientNetV2-S + ArcFace
- DINOv2 + GeM 또는 SALAD
- 서로 다른 resize 방식

현재 코드는 여러 checkpoint의 cosine score 평균을 이미 지원합니다. 검증셋에서 모델별
오답이 실제로 다른지 확인한 뒤 2~3개만 선택해야 합니다.

## 11. Pair별 지역 특징 검증

최종 문제는 각 Anchor에 대해 Positive 후보와 Negative 후보가 이미 주어집니다. 따라서
대규모 DB 전체를 지역 특징으로 검색할 필요 없이 두 쌍만 자세히 비교할 수 있습니다.

권장 2단계 점수는 다음과 같습니다.

```text
최종 점수 = α × global cosine + β × local geometric score
```

local score는 SuperPoint 같은 keypoint와 LightGlue matcher로 대응점을 찾고, RANSAC
inlier 수나 비율로 만들 수 있습니다. 같은 건물이라면 시점이 달라도 창문 모서리와
문양의 기하 구조가 일치할 가능성이 높습니다.

LightGlue는 문제 난이도에 따라 계산을 줄이는 local feature matcher이며, 논문에서
정확도와 효율 개선을 보고했습니다.

근거: [LightGlue, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)

주의할 점은 자연물, 야간, 계절 변화, 겹치는 영역이 작은 사진에서는 local match가
불안정할 수 있다는 것입니다. `α`, `β`는 한국 데이터가 아닌 검증셋에서 정해야 합니다.

## 12. 지금은 우선순위가 낮은 기술

### CosPlace

CosPlace는 classification 방식으로 대규모 geo-localization을 효율적으로 학습하고
domain 변화에도 강한 결과를 보고했습니다. 하지만 논문 자체가 작은 데이터나 방향
정보가 없는 데이터에는 적합하지 않다고 설명합니다. 현재 일부 GLDv2 CSV에는 위치와
orientation 정보가 없으므로 바로 적용할 1순위는 아닙니다.

근거: [CosPlace, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Berton_Rethinking_Visual_Geo-Localization_for_Large-Scale_Applications_CVPR_2022_paper.html)

### 단순히 embedding 차원만 크게 만들기

512를 1,024나 2,048로 늘리면 저장 공간과 계산량은 증가하지만 데이터가 부족하면
과적합만 커질 수 있습니다. 먼저 데이터 구성, loss, backbone을 개선한 뒤 실험합니다.

### 채점 이미지로 self-supervised 학습

label을 보지 않더라도 한국 채점 이미지를 모델 업데이트에 사용하면 “한국 랜드마크를
직접 학습하지 않는다”는 규칙을 위반한 것으로 해석될 수 있습니다. 하지 않는 것이
안전합니다.

## 13. 권장 실험 순서

한 번에 한 가지 조건만 바꿔야 어떤 기술이 효과가 있었는지 알 수 있습니다.

| 실험 | 바꾸는 것 | 나머지 조건 | 통과 기준 |
| --- | --- | --- | --- |
| D0 | 경로·label·국가·label당 장수 수정 | - | 데이터 audit 통과 |
| E0 | 현재 EfficientNetV2-S baseline | 고정 | 기준 Accuracy 기록 |
| E1 | ArcFace | E0와 동일 | Accuracy와 margin 개선 |
| E2 | Sub-center ArcFace | E1와 동일 | noisy label에서 개선 여부 |
| E3 | 384 입력 | 가장 좋은 loss 고정 | Accuracy 대비 시간 기록 |
| E4 | 약한 증강 | 해상도 고정 | 3개 seed 평균 개선 |
| E5 | DINOv2-S + GeM | 데이터/split 고정 | EfficientNet과 비교 |
| E6 | XBM 또는 SupCon | backbone 고정 | hard triplet 개선 |
| E7 | hard-negative 재탐색 | 가장 좋은 모델 고정 | 1회/2회 차이 확인 |
| E8 | TTA 네 종류 | checkpoint 고정 | 정확도와 추론시간 비교 |
| E9 | 2개 모델 ensemble | 단일 모델과 비교 | 오답 보완 여부 확인 |
| E10 | LightGlue score 결합 | global score 고정 | pair Accuracy 개선 |

검증 split 하나만 보면 우연에 속을 수 있습니다. 가능하면 비한국 label-disjoint split을
seed 3개로 만들고 평균과 표준편차를 기록합니다. 최종 실험 선택이 끝날 때까지 한국
채점 데이터의 결과를 보고 계속 튜닝하지 않습니다.

## 14. 가장 현실적인 추천 조합

현재 RTX 5070 한 장과 일부 GLDv2 데이터라는 조건에서 다음 조합을 먼저 권장합니다.

```text
외부 PC에서 한국을 제거한 현재의 일부 데이터
→ EfficientNetV2-S 또는 DINOv2-S
→ GeM + 512차원 projection
→ ArcFace + 약한 Triplet Loss
→ 384 fine-tuning
→ 주기적으로 갱신한 FAISS hard negatives
→ flip 또는 검증으로 선택한 multi-scale TTA
→ 성격이 다른 모델 2개 score ensemble
```

그 다음 계산 여유가 있을 때 `DINOv2 + SALAD` 또는 `global cosine + LightGlue local
score`를 시도하는 순서가 좋습니다.

## 15. 현재 제약을 반영한 최종 실행안

이미지 다운로드와 한국 이미지 필터링은 별도 PC에서 수행하고, 이 PC에는 완성된 일부
학습 이미지와 CSV를 옮긴다고 가정합니다. 아래 순서는 그 이후의 학습·추론 순서입니다.

### D0. 외부 준비 결과 확인

- 외부 PC에서 한국 랜드마크를 제거했는지 준비 로그를 보관합니다.
- CSV의 `path`가 이 PC의 `--image-root` 기준으로 열리는지 확인합니다.
- 실제 파일이 존재하는 행만 읽히는지 확인합니다.
- `label` 번호가 외부 PC에서 사용한 한국 제외 목록과 같은 체계인지 확인합니다.
- 외부 PC의 필터 로그와 CSV 해시를 함께 보관합니다.

이 모델은 이미지를 다운로드하거나 한국 여부를 판정하지 않습니다. 외부 PC가 데이터
준비를 담당하고, 이 PC는 준비된 결과를 검사하고 학습하는 구조입니다.

### E0. 현재 597개 label 기준선

현재 코드의 `min_images_per_label=2`를 유지한 채 baseline을 한 번 학습합니다. 이
결과를 모든 실험의 기준 Accuracy로 저장합니다. 사진이 적은 label을 sampler가 반복해
뽑는다는 점과 과적합 가능성을 함께 기록합니다.

### E1. ArcFace

일반 classifier를 cosine/Angular-margin classifier로 바꾸고, 기존 Triplet Loss를
작은 비율로 유지합니다. 먼저 `margin`만 0.2, 0.3, 0.4로 비교합니다.

### E2. singleton 활용 2단계 학습

코드 변경이 가능할 때만 진행합니다.

1. 모든 행을 사용하는 classification 사전학습
2. 2장 이상인 597개 label만 사용하는 metric fine-tuning

이 방식은 새 이미지를 받지 않고 singleton도 활용합니다. 외부 PC의 한국 제거 로그와
CSV 해시를 함께 보존하고, 이 PC에서 그 기록을 검증할 수 있어야 제출용 모델로 사용합니다.

### E3. 해상도·TTA·ensemble

ArcFace 또는 baseline 중 좋은 checkpoint를 고정하고 다음을 비교합니다.

```text
300 none → 300 flip → 300 five_crop_flip → 384 flip
```

그 뒤 EfficientNetV2와 DINOv2 같은 서로 다른 모델의 score ensemble을 시도합니다.
정확도뿐 아니라 추론 시간도 함께 기록합니다.

### E4. Hard Negative 보강

현재 모델로 FAISS hard negative를 만들고 1~2 epoch만 fine-tuning합니다. 이후 새
checkpoint로 다시 검색합니다. 처음부터 가장 어려운 후보만 사용하면 label 오류를
강하게 학습할 수 있으므로, 처음에는 hard 비율을 낮게 둡니다.

### E5. 지역 특징은 마지막

global embedding만으로 계속 틀리는 같은 모양의 건물 쌍에만 LightGlue 또는 SALAD를
적용합니다. 모든 pair에 local matching을 돌리면 추론 비용이 크게 늘 수 있으므로,
global score가 비슷한 상위 후보에만 재순위화하는 구조가 적절합니다.

## 16. 결론

현재 가장 큰 병목은 모델이 아니라 데이터입니다. 8,259장을 받았지만 label당 한 장인
경우가 너무 많아 대부분 학습에서 버려집니다. 같은 10GB라도 label당 여러 장을 확보하면
Triplet, SupCon, ArcFace가 제대로 작동할 수 있습니다.

그 다음 가장 먼저 적용할 모델 기술은 ArcFace입니다. 현재 cosine similarity 평가와
직접 맞고 GLDv2 공식 기준선에도 사용됐습니다. 이후 DINOv2, XBM, SALAD, LightGlue를
비용이 낮은 순서대로 검증하는 것이 좋습니다.

