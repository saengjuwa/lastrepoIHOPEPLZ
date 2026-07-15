# 글로벌 랜드마크 유사도 학습

한국 랜드마크를 학습에 넣지 않고, 두 사진이 같은 장소인지 cosine similarity로 비교하는 프로젝트입니다. 모델은 사진 한 장을 숫자 벡터인 임베딩으로 바꿉니다. 같은 장소의 벡터는 가깝게, 다른 장소의 벡터는 멀게 학습합니다.

## 가장 쉬운 실행 방법

Windows에서는 다음 파일을 실행합니다.

```bat
run_windows.bat
```

메뉴에서 다음 순서만 따르면 됩니다.

1. `3. 데이터셋 최상위 폴더 지정`
2. `4. 공식 GLDv2 일부 다운로드/필터/CSV 생성`
3. `5. 프리셋 학습`
4. `7. Triplet 점수 예측`
5. `8. 점수 Accuracy 확인`

Windows 앱은 세부 설정 수십 개를 묻지 않습니다. 자주 쓰는 조합을 프리셋으로 제공하고, 고급 실험은 아래의 한 줄 명령으로 실행합니다.

## 데이터 폴더 구조

데이터셋 최상위 폴더만 한 번 지정하면 코드에 절대경로를 하드코딩할 필요가 없습니다.

```text
데이터셋 최상위 폴더/
├─ gldv2/
│  ├─ korean_label_ids.txt
│  ├─ metadata/train.csv
│  ├─ metadata/md5/train/
│  ├─ archives/train/images_000.tar ...
│  ├─ train/a/b/c/<image_id>.jpg
│  ├─ train_labels.csv
│  └─ preparation_audit.json
└─ data/
   ├─ triplets.json
   └─ validation/
```

- `gldv2`는 비한국 학습 데이터입니다.
- `data`는 한국 채점 데이터입니다. 학습, hard-negative 생성, fine-tuning에 절대 넣지 않습니다.
- 생성된 `train_labels.csv`의 `path`는 `gldv2/train`을 기준으로 한 상대경로입니다. 다른 컴퓨터로 폴더를 옮겨도 그대로 열립니다.

## `korean_label_ids.txt`는 무엇인가요?

아주 쉽게 말하면 “학습 금지 명단”입니다.

GLDv2 공식 `train.csv`에는 이미지 ID, URL, landmark ID만 있고 나라 정보가 없습니다. 그래서 프로그램은 어떤 landmark가 한국인지 혼자 알아낼 수 없습니다. 외부 컴퓨터에서 사용 중인 신뢰 가능한 한국 필터 결과를 다음처럼 한 줄에 하나씩 저장해야 합니다.

```text
# korean_label_ids.txt
2050
8162
10042
```

이 파일은 두 번 사용됩니다.

1. 다운로드한 TAR에서 이미지를 꺼낼 때 이 label의 이미지를 제외합니다.
2. 학습을 시작할 때 CSV에 이 label이 남지 않았는지 다시 검사합니다.

빈 파일은 “한국 label이 하나도 없다”는 증거가 아니므로 거부됩니다. 이 목록을 만들 수 없다면 프로그램은 규칙을 지켰다고 증명할 수 없으므로 학습하지 않는 것이 맞습니다. 한국 채점 이미지로 목록을 만들거나 모델을 튜닝해서도 안 됩니다.

## 공식 GLDv2 일부 다운로드

데이터는 [cvdfoundation/google-landmark](https://github.com/cvdfoundation/google-landmark)의 공식 S3 형식을 사용합니다. 학습 TAR는 `images_000.tar`부터 `images_499.tar`까지 500개이고, shard 하나는 약 1GB입니다.

다음 명령은 처음 10개 shard를 받습니다. `10`을 원하는 개수로 바꿀 수 있으며 유효 범위는 1~500입니다.

```powershell
python scripts\prepare_gldv2.py --dataset-root "D:\datasets" --archive-count 10 --korean-labels-file "D:\datasets\gldv2\korean_label_ids.txt" --min-images-per-label 1
```

한국 label 파일은 처음 실행할 때 HuggingFace `visheratin/google_landmarks_places`에서 `country`가 `South Korea` 또는 `North Korea`인 행의 `id`를 모아 `gldv2/korean_label_ids.txt`로 자동 생성합니다. 기존 파일은 재사용하고, 다시 받으려면 `--refresh-korean-labels`를 추가합니다.

다운로더가 하는 일은 다음과 같습니다.

- 공식 `train.csv`, TAR, MD5 파일을 받습니다.
- `.part` 파일을 사용해 덜 받은 파일을 완성본으로 착각하지 않습니다.
- MD5가 맞는 TAR만 사용하며, 다시 실행하면 검증된 파일을 재사용합니다.
- TAR 안의 경로를 그대로 믿지 않고 안전한 출력 경로를 직접 만듭니다.
- 한국 label을 제외합니다. `--min-images-per-label 1`이므로 한 장뿐인 label도 CSV에 남고, 일반 metric 학습은 기본값 2로 다시 거릅니다.
- `gldv2/train_labels.csv`와 `preparation_audit.json`을 만듭니다.

TAR와 추출된 JPEG를 함께 보관하므로 shard 개수보다 더 많은 디스크 공간이 필요합니다. TAR 안에는 여러 나라가 섞여 있어 한국 bytes 자체의 다운로드를 미리 막을 수는 없고, 검증된 한국 label을 추출·manifest 단계에서 제외합니다.

## 설치와 확인

다음 명령은 모두 한 줄입니다.

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m unittest discover -s tests -v
```

GPU 이름은 Windows 앱 메뉴 2 또는 다음 명령으로 확인합니다. RTX 5070이라는 문자열을 코드에 고정하지 않고 드라이버가 실제로 보고한 이름을 출력합니다.

```powershell
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

## 왜 예전 여러 줄 명령이 실패했나요?

사용자 제보는 맞습니다. PowerShell의 backtick(`)은 줄의 맨 마지막 글자일 때만 다음 줄을 이어 줍니다. 뒤에 보이지 않는 공백이 있거나 CMD에서 실행하면 명령이 중간에 끊깁니다. 이 README의 실행 명령은 모두 한 물리적 줄로 바꿨습니다. Windows 앱도 문자열을 줄바꿈으로 실행하지 않고 인수 목록을 `subprocess`에 직접 전달하므로 같은 문제가 없습니다.

## 학습 프리셋 명령

아래 예시는 `$datasetRoot`를 먼저 지정한 PowerShell 기준입니다.

```powershell
$datasetRoot = "D:\datasets"
```

### 1. EfficientNetV2-S + GeM + ArcFace

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --arcface-scale 30 --arcface-margin 0.3 --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\efficientnet_arcface_best.pt
```

### 2. Sub-center ArcFace

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier subcenter_arcface --subcenters 3 --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\efficientnet_subcenter_best.pt
```

class마다 중심을 3개 만들므로 noisy label에 도움이 될 수 있지만 class 수가 아주 많으면 classifier 메모리도 약 3배가 됩니다.

### 3. singleton까지 사용하는 분류 사전학습

1장뿐인 label은 Positive 쌍을 만들 수 없지만 class 번호를 맞히는 분류 학습에는 쓸 수 있습니다. 첫 단계에서는 모든 label을 사용하고, 두 번째 metric 단계에서는 2장 이상인 label만 사용합니다. label 수가 바뀌면 classifier는 다시 만들고 backbone과 projection만 가져옵니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 300 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --training-stage classification --metric-loss none --min-images-per-label 1 --augmentation weak --epochs 3 --batch-size 32 --output checkpoints\classification_pretrain.pt
```

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --init-checkpoint checkpoints\classification_pretrain.pt --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\metric_after_classification.pt
```

### 4. DINOv2-S + token GeM

DINOv2의 patch 크기는 14입니다. 기본 프리셋은 384 대신 14로 나누어떨어지는 378을 사용합니다. 현재 구현은 CLS token과 patch token을 분리하고 patch에 실제 GeM을 적용합니다. DINOv2에는 ImageNet mean/std를 사용합니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone dinov2_small --image-size 378 --freeze-backbone --train-last-blocks 2 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 8 --labels-per-batch 4 --images-per-label 2 --output checkpoints\dinov2_gem_best.pt
```

### 5. DINOv2-S + compact SALAD

SALAD는 patch를 cluster에 배정하고 배경용 dustbin을 둔 뒤 지역 특징과 전역 특징을 합칩니다. 공식 GPL 코드를 복사하지 않고 논문의 Sinkhorn+dustbin 구조를 독립 구현했습니다. RTX 5070에서 시작하기 쉽게 `16 cluster × 64 local + 256 global` 뒤 512 projection을 쓰는 compact 설정입니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone dinov2_small --image-size 378 --freeze-backbone --train-last-blocks 2 --pooling salad --salad-clusters 16 --salad-local-dim 64 --salad-global-dim 256 --sinkhorn-iterations 3 --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 8 --labels-per-batch 4 --images-per-label 2 --output checkpoints\dinov2_salad_best.pt
```

### 6. DOLG-style 지역·전역 결합

이 옵션은 spatial attention 지역 특징에서 전역 방향 성분을 제거한 뒤 전역 특징과 합치는 compact DOLG-style head입니다. 원 논문의 다중 dilation 전체 구조를 그대로 복제한 것은 아니므로 결과 자료에서도 `DOLG-style`이라고 표기해야 합니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling dolg --dolg-dim 512 --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\efficientnet_dolg_best.pt
```

## 다른 metric loss와 XBM

한 번에 모두 켜지 말고 같은 split에서 하나씩 비교합니다.

SupCon은 같은 label의 나머지 사진을 모두 Positive로 씁니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --classifier arcface --metric-loss supcon --supcon-temperature 0.07 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\supcon.pt
```

Proxy Anchor는 class별 별도 proxy를 학습합니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --classifier linear --metric-loss proxy_anchor --proxy-alpha 32 --proxy-margin 0.1 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\proxy_anchor.pt
```

XBM은 이전 batch 임베딩을 FIFO 메모리에 저장합니다. checkpoint에는 queue도 함께 저장됩니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --classifier arcface --metric-loss triplet --xbm-size 4096 --xbm-weight 0.2 --xbm-warmup-steps 100 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\arcface_xbm.pt
```

## FAISS hard-negative 갱신

먼저 현재 checkpoint로 다른 label 중 가까운 사진을 찾습니다.

```powershell
python scripts\mine_hard_negatives.py --checkpoint checkpoints\efficientnet_arcface_best.pt --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --split-manifest outputs\landmark_split.json --index-type hnsw --top-k 20 --save-index outputs\hard_negatives.faiss --output outputs\hard_negatives.csv
```

그다음 초반 20%에서 후반 70%까지 hard batch 비율을 올리며 짧게 fine-tuning합니다.

```powershell
python scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --init-checkpoint checkpoints\efficientnet_arcface_best.pt --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --hard-negatives-csv outputs\hard_negatives.csv --select-best-triplet --split-manifest outputs\landmark_split.json --hard-negative-ratio-start 0.2 --hard-negative-ratio 0.7 --hard-negative-weight 0.5 --epochs 2 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\hard_finetuned.pt
```

모델이 바뀌면 어려운 Negative도 바뀌므로 새 checkpoint로 mining을 다시 실행할 수 있습니다. 한국 평가 이미지는 mining 대상이 아닙니다.

## 추론: TTA, multi-scale, ensemble, LightGlue

`five_crop_flip`은 5개 crop과 좌우 반전을 함께 써서 이미지당 10개 view를 평균냅니다. `--scales`는 크기별 임베딩을 다시 평균하고 L2 정규화합니다.

```powershell
python scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta five_crop_flip --scales 300,384 --output outputs\scores_tta.csv
```

서로 다른 checkpoint는 임베딩 차원이 달라도 각 모델의 cosine 점수를 평균해 ensemble합니다. DINOv2 scale은 14의 배수를 사용합니다.

```powershell
python scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt checkpoints\dinov2_salad_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta flip --output outputs\scores_ensemble.csv
```

LightGlue는 global 점수 차이가 기본 0.05 이하인 애매한 triplet만 지역 특징과 homography inlier ratio로 다시 확인합니다. 기본 설치에는 포함되지 않습니다.

```powershell
python -m pip install -r requirements-lightglue.txt
```

```powershell
python scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta flip --local-reranker lightglue --local-features aliked --local-weight 0.05 --local-margin-threshold 0.05 --output outputs\scores_lightglue.csv
```

Accuracy는 문제 정의 그대로 `sim_anchor_positive > sim_anchor_negative`인 비율입니다.

```powershell
python scripts\evaluate_scores.py --scores outputs\scores_lightglue.csv
```

## 실험 원칙

- 한국 채점 데이터로 모델을 업데이트하거나 설정을 계속 고르지 않습니다.
- 비한국 고정 split에서 한 번에 기술 하나만 바꿉니다.
- Accuracy뿐 아니라 평균 margin, 실행 시간, GPU 메모리도 기록합니다.
- seed 3개 평균을 보고 우연한 개선을 구분합니다.
- 자세한 기술 배경과 우선순위는 [PERFORMANCE_IMPROVEMENT_IDEAS_KO.md](PERFORMANCE_IMPROVEMENT_IDEAS_KO.md)를 참고합니다.
