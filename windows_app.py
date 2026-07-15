from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG_PATH = ROOT / ".windows_app_config.json"


TRAIN_PRESETS: dict[str, dict[str, object]] = {
    "1": {
        "name": "EfficientNetV2-S + GeM + ArcFace",
        "output": "checkpoints\\efficientnet_arcface_best.pt",
        "batch": 16,
        "options": [
            "--backbone", "efficientnetv2_s", "--image-size", "384",
            "--pooling", "gem", "--classifier", "arcface",
            "--metric-loss", "triplet", "--augmentation", "weak",
        ],
    },
    "2": {
        "name": "EfficientNetV2-S + Sub-center ArcFace",
        "output": "checkpoints\\efficientnet_subcenter_best.pt",
        "batch": 16,
        "options": [
            "--backbone", "efficientnetv2_s", "--image-size", "384",
            "--pooling", "gem", "--classifier", "subcenter_arcface",
            "--subcenters", "3", "--metric-loss", "triplet",
            "--augmentation", "weak",
        ],
    },
    "3": {
        "name": "DINOv2-S + token GeM + ArcFace",
        "output": "checkpoints\\dinov2_gem_best.pt",
        "batch": 8,
        "options": [
            "--backbone", "dinov2_small", "--image-size", "378",
            "--pooling", "gem", "--classifier", "arcface",
            "--metric-loss", "triplet", "--freeze-backbone",
            "--train-last-blocks", "2", "--augmentation", "weak",
        ],
    },
    "4": {
        "name": "DINOv2-S + compact SALAD + ArcFace",
        "output": "checkpoints\\dinov2_salad_best.pt",
        "batch": 8,
        "options": [
            "--backbone", "dinov2_small", "--image-size", "378",
            "--pooling", "salad", "--classifier", "arcface",
            "--metric-loss", "triplet", "--freeze-backbone",
            "--train-last-blocks", "2", "--augmentation", "weak",
        ],
    },
    "5": {
        "name": "EfficientNetV2-S + DOLG-style fusion + ArcFace",
        "output": "checkpoints\\efficientnet_dolg_best.pt",
        "batch": 16,
        "options": [
            "--backbone", "efficientnetv2_s", "--image-size", "384",
            "--pooling", "dolg", "--classifier", "arcface",
            "--metric-loss", "triplet", "--augmentation", "weak",
        ],
    },
    "6": {
        "name": "Singleton classification pretraining",
        "output": "checkpoints\\classification_pretrain.pt",
        "batch": 32,
        "classification": True,
        "options": [
            "--backbone", "efficientnetv2_s", "--image-size", "300",
            "--pooling", "gem", "--classifier", "arcface",
            "--training-stage", "classification", "--metric-loss", "none",
            "--min-images-per-label", "1", "--augmentation", "weak",
        ],
    },
}


def load_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def save_config(config: dict[str, str]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolved_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def default_dataset_root() -> str:
    configured = load_config().get("dataset_root", "").strip()
    if configured:
        return configured
    for candidate in (
        ROOT / "datasets",
        ROOT.parent / "datasets",
        ROOT.parent.parent / "datasets",
        ROOT / "data",
    ):
        if candidate.is_dir() and any(
            (candidate / name).exists() for name in ("gldv2", "data")
        ):
            return str(candidate)
    return ""


def dataset_path(*parts: str) -> str:
    root = default_dataset_root()
    return str((Path(root) if root else ROOT).joinpath(*parts))


def training_csv_status(value: str) -> tuple[str, set[str]]:
    path = resolved_path(value)
    if not path.is_file():
        return "missing", set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            columns = {item.strip() for item in next(csv.reader(file), [])}
    except (OSError, UnicodeError, csv.Error):
        return "unreadable", set()
    if {"id", "url", "landmark_id"}.issubset(columns):
        return ("gldv2_metadata_partial" if path.suffix == ".part" else "gldv2_metadata"), columns
    if path.suffix == ".part":
        return "partial", columns
    if {"path", "label"}.issubset(columns):
        return ("ready_with_country" if "country_code" in columns else "ready_needs_korean_labels"), columns
    return "wrong_columns", columns


def default_training_csv() -> str:
    candidates = [
        dataset_path("gldv2", "train_labels.csv"),
        dataset_path("gldv2", "train_10gb_labels.csv"),
    ]
    for candidate in candidates:
        if training_csv_status(candidate)[0].startswith("ready_"):
            return candidate
    return candidates[0]


def default_image_root() -> str:
    return dataset_path("gldv2", "train")


def default_korean_labels_file() -> str:
    return dataset_path("gldv2", "korean_label_ids.txt")


def default_checkpoint() -> str:
    return load_config().get("best_checkpoint", "checkpoints\\efficientnet_arcface_best.pt")


def ask_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip().strip('"')
    return value or default


def ask_int(label: str, default: int) -> int:
    while True:
        try:
            return int(ask_text(label, str(default)))
        except ValueError:
            print("정수를 입력하세요.")


def ask_yes_no(label: str, default: bool = False) -> bool:
    prompt = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{prompt}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("y 또는 n을 입력하세요.")


def script_path(name: str) -> str:
    return str(ROOT / "scripts" / name)


def run_command(args: list[str]) -> bool:
    print("\n실행할 한 줄 명령:")
    print(subprocess.list2cmdline(args))
    print()
    try:
        subprocess.run(args, cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"명령 실행 실패: {exc}")
        input("\nEnter를 누르면 메뉴로 돌아갑니다.")
        return False
    input("\nEnter를 누르면 메뉴로 돌아갑니다.")
    return True


def require_dataset_root() -> Path | None:
    value = default_dataset_root()
    if not value or not resolved_path(value).is_dir():
        print("먼저 메뉴 3에서 데이터셋 최상위 폴더를 지정하세요.")
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return None
    return resolved_path(value)


def explain_training_csv(value: str) -> tuple[bool, bool]:
    status, columns = training_csv_status(value)
    if status == "ready_with_country":
        print("학습 CSV 준비 완료: country_code=KR 행은 자동 제외됩니다.")
        return True, False
    if status == "ready_needs_korean_labels":
        print("학습 CSV에는 country_code가 없으므로 한국 label ID 파일이 필요합니다.")
        return True, True
    if status.startswith("gldv2_metadata"):
        print("이것은 공식 id,url,landmark_id 메타데이터이며 학습 CSV가 아닙니다.")
        print("먼저 메뉴 4로 일부 shard를 준비하세요.")
    else:
        shown = ", ".join(sorted(columns)) or "없음"
        print(f"path,label 학습 CSV를 읽을 수 없습니다. 발견한 열: {shown}")
    return False, False


def install_requirements() -> None:
    run_command([PYTHON, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def check_gpu() -> None:
    print("\nNVIDIA 드라이버가 실제로 보고한 GPU:")
    try:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi 실행 실패: {exc}")
    print("\n현재 Python/PyTorch가 사용할 GPU:")
    subprocess.run(
        [
            PYTHON,
            "-c",
            "import importlib.util; print('PyTorch not installed' if importlib.util.find_spec('torch') is None else (__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CUDA unavailable'))",
        ],
        cwd=ROOT,
        check=False,
    )
    input("\nEnter를 누르면 메뉴로 돌아갑니다.")


def set_dataset_root() -> None:
    value = ask_text("데이터셋 최상위 폴더의 전체 경로", default_dataset_root())
    if not value:
        return
    path = resolved_path(value)
    if not path.is_dir():
        print(f"폴더가 없습니다: {path}")
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return
    config = load_config()
    config["dataset_root"] = str(path)
    save_config(config)
    print(f"저장됨: {path}")
    print(f"학습 기본 위치: {path / 'gldv2'}")
    print(f"평가 기본 위치: {path / 'data'}")
    input("Enter를 누르면 메뉴로 돌아갑니다.")


def prepare_gldv2() -> None:
    root = require_dataset_root()
    if root is None:
        return
    print("\n공식 GLDv2 학습 shard 일부 다운로드/준비")
    print("shard 1개는 약 1GB입니다. TAR와 추출 이미지가 함께 있어 추가 공간이 필요합니다.")
    archive_count = ask_int("처음부터 받을 shard 개수 (1~500)", 10)
    while not 1 <= archive_count <= 500:
        print("shard 개수는 1부터 500 사이여야 합니다.")
        archive_count = ask_int(
            "처음부터 받을 shard 개수 (1~500)", 10
        )
    korean_file = ask_text("한국 landmark_id 목록 파일 (없으면 자동 생성)", default_korean_labels_file())
    print("Korean label file is missing; HuggingFace labels will be generated automatically.")
    run_command(
        [
            PYTHON,
            script_path("prepare_gldv2.py"),
            "--dataset-root", str(root),
            "--archive-count", str(archive_count),
            "--korean-labels-file", korean_file,
            "--min-images-per-label", "1",
        ]
    )


def choose_preset() -> tuple[str, dict[str, object]] | None:
    print("\n학습 프리셋")
    for key, preset in TRAIN_PRESETS.items():
        print(f"{key}. {preset['name']}")
    choice = ask_text("선택", "1")
    preset = TRAIN_PRESETS.get(choice)
    if preset is None:
        print("없는 프리셋입니다.")
        return None
    return choice, preset


def train_preset() -> None:
    if require_dataset_root() is None:
        return
    selected = choose_preset()
    if selected is None:
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return
    _, preset = selected
    csv_path = ask_text("학습 CSV", default_training_csv())
    ready, needs_korean = explain_training_csv(csv_path)
    if not ready:
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return
    image_root = ask_text("학습 이미지 폴더", default_image_root())
    korean_file = ""
    if needs_korean:
        korean_file = ask_text("한국 landmark_id 목록 파일", default_korean_labels_file())
        if not resolved_path(korean_file).is_file():
            print("한국 label ID 파일이 없어 학습을 중단합니다.")
            input("Enter를 누르면 메뉴로 돌아갑니다.")
            return
    epochs = ask_int("epoch 수", 5)
    batch = ask_int("batch 크기", int(preset["batch"]))
    if not preset.get("classification") and (batch < 4 or batch % 2):
        print("metric 프리셋의 batch는 4 이상의 짝수여야 합니다.")
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return
    output = ask_text("checkpoint 출력", str(preset["output"]))
    args = [
        PYTHON, script_path("train.py"), "--csv", csv_path,
        "--image-root", image_root, "--pretrained", "--use-projection",
        "--embedding-dim", "512", "--epochs", str(epochs),
        "--batch-size", str(batch), "--num-workers", "8",
        "--resize-mode", "center_crop", "--output", output,
    ]
    args.extend(str(value) for value in preset["options"])
    if not preset.get("classification"):
        args.extend(
            [
                "--labels-per-batch", str(batch // 2),
                "--images-per-label", "2", "--select-best-triplet",
                "--split-manifest", "outputs\\landmark_split.json",
                "--hard-val-fraction", "0.5", "--val-tta", "flip",
            ]
        )
    if korean_file:
        args.extend(["--korean-labels-file", korean_file])
    if run_command(args):
        config = load_config()
        config["best_checkpoint"] = output
        save_config(config)


def mine_hard_negatives() -> None:
    if require_dataset_root() is None:
        return
    checkpoint = ask_text("checkpoint", default_checkpoint())
    csv_path = ask_text("학습 CSV", default_training_csv())
    ready, needs_korean = explain_training_csv(csv_path)
    if not ready:
        input("Enter를 누르면 메뉴로 돌아갑니다.")
        return
    korean_file = ask_text("한국 landmark_id 목록 파일", default_korean_labels_file()) if needs_korean else ""
    output = ask_text("hard-negative CSV 출력", "outputs\\hard_negatives.csv")
    args = [
        PYTHON, script_path("mine_hard_negatives.py"),
        "--checkpoint", checkpoint, "--csv", csv_path,
        "--image-root", default_image_root(),
        "--split-manifest", "outputs\\landmark_split.json",
        "--index-type", "hnsw", "--top-k", "20", "--output", output,
    ]
    if korean_file:
        args.extend(["--korean-labels-file", korean_file])
    run_command(args)


def predict() -> None:
    if require_dataset_root() is None:
        return
    raw_checkpoints = ask_text("checkpoint 경로(ensemble은 ; 로 구분)", default_checkpoint())
    checkpoints = [item.strip().strip('"') for item in raw_checkpoints.split(";") if item.strip()]
    if not checkpoints:
        print("checkpoint를 하나 이상 입력하세요.")
        return
    tta_choices = {"1": "none", "2": "flip", "3": "five_crop_flip"}
    print("1. TTA 없음  2. flip  3. five-crop + flip(10 view)")
    tta = tta_choices.get(ask_text("TTA", "2"), "flip")
    scales = ask_text("다중 해상도(예: 300,384, 비우면 checkpoint 크기)", "")
    output = ask_text("점수 CSV 출력", "outputs\\scores.csv")
    args = [
        PYTHON, script_path("predict_triplets.py"), "--checkpoint", *checkpoints,
        "--triplets", dataset_path("data", "triplets.json"),
        "--image-root", dataset_path("data", "validation"),
        "--tta", tta, "--output", output,
    ]
    if scales:
        args.extend(["--scales", scales])
    if ask_yes_no("점수가 비슷한 pair에 LightGlue를 사용합니까?", False):
        args.extend(["--local-reranker", "lightglue"])
    run_command(args)


def evaluate() -> None:
    scores = ask_text("점수 CSV", "outputs\\scores.csv")
    run_command([PYTHON, script_path("evaluate_scores.py"), "--scores", scores])


def print_menu() -> None:
    root = default_dataset_root() or "미설정(메뉴 3)"
    print("\n랜드마크 학습 도우미")
    print(f"데이터셋 최상위 폴더: {root}")
    print(f"기본 학습 CSV: {default_training_csv()}")
    print("1. 필수 패키지 설치")
    print("2. 실제 GPU 확인")
    print("3. 데이터셋 최상위 폴더 지정")
    print("4. 공식 GLDv2 일부 다운로드/필터/CSV 생성")
    print("5. 프리셋 학습")
    print("6. FAISS hard-negative 생성")
    print("7. Triplet 점수 예측")
    print("8. 점수 Accuracy 확인")
    print("0. 종료")


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows landmark training helper")
    parser.add_argument("--list-actions", action="store_true")
    args = parser.parse_args()
    if args.list_actions:
        print_menu()
        return
    actions = {
        "1": install_requirements,
        "2": check_gpu,
        "3": set_dataset_root,
        "4": prepare_gldv2,
        "5": train_preset,
        "6": mine_hard_negatives,
        "7": predict,
        "8": evaluate,
    }
    while True:
        print_menu()
        choice = input("선택: ").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if action is None:
            print("없는 메뉴입니다.")
        else:
            action()


if __name__ == "__main__":
    main()
