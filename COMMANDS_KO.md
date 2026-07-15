# 랜드마크 학습·추론 명령어 모음

이 문서의 명령은 모두 PowerShell에서 한 줄 그대로 실행합니다. 명령 중간에 임의로 줄을 바꾸지 마세요.

## 1. 폴더 이동과 기본 변수

프로그램 폴더와 데이터 폴더는 서로 다릅니다.

```powershell
Set-Location "D:\smth\coding\ogq\ft_llm_edited"
```

```powershell
$datasetRoot = "D:\smth\coding\datasets"
```

<<<<<<< HEAD
=======
처음 다운로드할 때 `korean_label_ids.txt`가 없으면 HuggingFace에서 자동 생성됩니다. 다시 생성하려면 다운로드 명령 끝에 `--refresh-korean-labels`를 붙이세요.

>>>>>>> 4a21bd9 (bloom into you)
폴더 구조는 다음과 같아야 합니다.

```text
D:\smth\coding\datasets\gldv2\train_labels.csv
D:\smth\coding\datasets\gldv2\train\
D:\smth\coding\datasets\gldv2\korean_label_ids.txt
D:\smth\coding\datasets\data\triplets.json
D:\smth\coding\datasets\data\validation\
```

## 2. 설치와 GPU 확인

```powershell
py -3 -m pip install -r requirements.txt
```

```powershell
nvidia-smi
```

프로그램이 PyTorch에서 보는 GPU 이름을 확인합니다.

```powershell
py -3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## 3. GLDv2 데이터 다운로드·준비

`--archive-count 10`은 처음 10개 shard를 받는다는 뜻입니다. 1개 shard는 약 1GB입니다.

```powershell
py -3 scripts\prepare_gldv2.py --dataset-root "$datasetRoot" --archive-count 10 --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --min-images-per-label 1
```

다운로더는 한국 label을 제외하고, `train_labels.csv`와 `preparation_audit.json`을 만듭니다. 한국 label 파일은 비어 있으면 안 됩니다.

## 4. 가장 쉬운 학습: EfficientNetV2 + GeM + ArcFace

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --arcface-scale 30 --arcface-margin 0.3 --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\efficientnet_arcface_best.pt
```

## 5. Sub-center ArcFace

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier subcenter_arcface --subcenters 3 --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\efficientnet_subcenter_best.pt
```

## 6. Singleton classification 사전학습 후 metric 학습

사진이 한 장뿐인 label도 먼저 분류 학습에 사용할 수 있습니다.

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 300 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --training-stage classification --metric-loss none --min-images-per-label 1 --augmentation weak --epochs 3 --batch-size 32 --output checkpoints\classification_pretrain.pt
```

그 checkpoint를 사용해 같은 landmark끼리 가까워지는 metric 학습을 합니다.

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --init-checkpoint checkpoints\classification_pretrain.pt --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\metric_after_classification.pt
```

## 7. DINOv2와 SALAD

DINOv2 입력 크기는 14의 배수여야 합니다. 378을 사용합니다.

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone dinov2_small --image-size 378 --freeze-backbone --train-last-blocks 2 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 8 --labels-per-batch 4 --images-per-label 2 --output checkpoints\dinov2_gem_best.pt
```

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone dinov2_small --image-size 378 --freeze-backbone --train-last-blocks 2 --pooling salad --salad-clusters 16 --salad-local-dim 64 --salad-global-dim 256 --sinkhorn-iterations 3 --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --augmentation weak --select-best-triplet --split-manifest outputs\landmark_split.json --hard-val-fraction 0.5 --val-tta flip --epochs 5 --batch-size 8 --labels-per-batch 4 --images-per-label 2 --output checkpoints\dinov2_salad_best.pt
```

## 8. SupCon, ProxyAnchor, XBM

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss supcon --supcon-temperature 0.07 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\supcon.pt
```

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier linear --metric-loss proxy_anchor --proxy-alpha 32 --proxy-margin 0.1 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\proxy_anchor.pt
```

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --pretrained --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --xbm-size 4096 --xbm-weight 0.2 --xbm-warmup-steps 100 --augmentation weak --epochs 5 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\arcface_xbm.pt
```

## 9. Hard-negative mining과 재학습

먼저 현재 checkpoint로 어려운 Negative를 찾습니다.

```powershell
py -3 scripts\mine_hard_negatives.py --checkpoint checkpoints\efficientnet_arcface_best.pt --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --split-manifest outputs\landmark_split.json --index-type hnsw --top-k 20 --save-index outputs\hard_negatives.faiss --output outputs\hard_negatives.csv
```

그 결과를 사용해 hard-negative 비율을 20%에서 70%까지 올리며 재학습합니다.

```powershell
py -3 scripts\train.py --csv "$datasetRoot\gldv2\train_labels.csv" --image-root "$datasetRoot\gldv2\train" --korean-labels-file "$datasetRoot\gldv2\korean_label_ids.txt" --init-checkpoint checkpoints\efficientnet_arcface_best.pt --backbone efficientnetv2_s --image-size 384 --pooling gem --use-projection --embedding-dim 512 --classifier arcface --metric-loss triplet --hard-negatives-csv outputs\hard_negatives.csv --select-best-triplet --split-manifest outputs\landmark_split.json --hard-negative-ratio-start 0.2 --hard-negative-ratio 0.7 --hard-negative-weight 0.5 --epochs 2 --batch-size 16 --labels-per-batch 8 --images-per-label 2 --output checkpoints\hard_finetuned.pt
```

## 10. Triplet 추론

기본 flip TTA입니다.

```powershell
py -3 scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta flip --output outputs\scores.csv
```

5-crop과 flip을 동시에 사용하고, 300과 384 두 크기를 함께 사용합니다.

```powershell
py -3 scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta five_crop_flip --scales 300,384 --output outputs\scores_tta.csv
```

두 checkpoint의 결과를 ensemble합니다.

```powershell
py -3 scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt checkpoints\dinov2_salad_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta flip --output outputs\scores_ensemble.csv
```

LightGlue는 애매한 triplet만 다시 확인합니다. 먼저 선택 의존성을 설치합니다.

```powershell
py -3 -m pip install -r requirements-lightglue.txt
```

```powershell
py -3 scripts\predict_triplets.py --checkpoint checkpoints\efficientnet_arcface_best.pt --triplets "$datasetRoot\data\triplets.json" --image-root "$datasetRoot\data\validation" --tta flip --local-reranker lightglue --local-features aliked --local-weight 0.05 --local-margin-threshold 0.05 --output outputs\scores_lightglue.csv
```

## 11. Accuracy 확인

```powershell
py -3 scripts\evaluate_scores.py --scores outputs\scores.csv
```

## 12. Windows 앱으로 실행할 때

명령어를 직접 입력하기 어렵다면 다음을 실행합니다.

```powershell
Set-Location "D:\smth\coding\ogq\ft_llm_edited"
py -3 windows_app.py
```

앱 메뉴 순서는 보통 다음과 같습니다.

1. GPU 확인
2. 데이터 루트 지정: `D:\smth\coding\datasets`
3. GLDv2 다운로드·필터링
4. 학습 preset 선택
5. Triplet 추론
6. Accuracy 확인

처음에는 EfficientNetV2 + GeM + ArcFace preset으로 시작한 뒤, 성능 비교가 필요할 때 DINOv2, SALAD, SupCon, XBM을 하나씩 시험하는 것이 좋습니다.
