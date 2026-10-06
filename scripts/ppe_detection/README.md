# PPE detection (hardhat / safety vest) with D435i

Per-person check of hardhat and hi-vis vest using two YOLO models and D435i depth.

```
person 1 (2.1m) : 안전모 OK  안전복 NO
```

## How it works
1. D435i color + depth (640x480 @ 30 fps), depth aligned to color.
2. `models/best.pt` (PPE model) detects `Hardhat` / `Safety Vest` boxes.
3. `models/yolo11s.pt` (COCO) detects people; ByteTrack assigns IDs.
4. Per person: hardhat in the top 0-35 % band, vest in the 20-75 % band of the person box
   (>= 50 % of the PPE box inside), majority vote over the last 15 frames.
5. Distance = median depth of the upper-torso patch; people outside 0.5-5 m are skipped.

## Setup
```bash
cd scripts/ppe_detection
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
mkdir -p models
.venv/bin/python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('Hexmon/vyra-yolo-ppe-detection', 'best.pt'), 'models/best.pt')"
```
`models/yolo11s.pt` is downloaded automatically by ultralytics on first run if missing
(it lands in the working directory; move it to `models/`).

## Run
```bash
.venv/bin/python main.py            # q to quit
.venv/bin/python main.py --conf 0.25 --min-dist 1 --max-dist 4
.venv/bin/python test_match.py      # matching logic self-check
```

## Status
- Hardhat detected at ~1.3-1.4 m in the first live test; vest rarely detected; track IDs churn.
- Next: compare other PPE models, add hi-vis color check, fine-tune on own footage.

## Models / licenses
- PPE: [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) (YOLOv8m, CC BY 4.0)
- Person: Ultralytics YOLO11s (AGPL-3.0)

Weights are not committed (`*.pt` is git-ignored).
