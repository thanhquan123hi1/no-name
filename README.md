# BiasLoraAsy Deepfake Detector

This repository is a cleaned research project for the `BiasLoraAsy` detector.
It keeps the DeepfakeBench-style training/evaluation pipeline, but removes the
extra detector implementations and configs that are not needed for this model.

## Model

`BiasLoraAsy` uses a CLIP ViT-L/14 vision backbone with LoRA adapters on the
last attention blocks, trainable backbone biases, a binary classifier head, and
an asymmetric supervised contrastive loss.

Main files:

- `training/detectors/BiasLoraAsy.py`
- `training/config/detector/BiasLoraAsy.yaml`
- `training/train.py`
- `training/test.py`

## Setup

Install the Python dependencies used by the training pipeline:

```bash
pip install -r requirements.txt
```

Place frame data and dataset JSON files wherever you prefer, then update:

- `rgb_dir`
- `dataset_json_folder`
- `log_dir`

in `training/config/train_config.yaml` and `training/config/test_config.yaml`.

The dataset loader expects DeepfakeBench-style JSON metadata, for example:

```text
preprocessing/dataset_json/FaceForensics++.json
datasets/rgb/<dataset>/.../frames/*.png
```

## Training

```bash
python training/train.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2"
```

Train the bias-only CLIP detector with weak-to-strong consistency:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasConsistency.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0"
```

`BiasConsistency` trains only CLIP vision-backbone parameters whose names end
in `.bias`, plus the binary classifier. Its objective is the mean
cross-entropy of weak/strong views plus a scheduled KL divergence from the
detached weak-view prediction to the strong-view prediction. Evaluation uses a
single image view.

Train the artifact-preserving variant:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasArtifactConsistency.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "Celeb-DF-v2"
```

`BiasArtifactConsistency` keeps bias-only CLIP adaptation, but restricts KL to
confident weak predictions that agree with the ground-truth label. It uses
class-balanced consistency and sampling, gives the artifact-preserving weak
view more CE weight, and uses a milder strong augmentation policy.

Optional fine-tuning from a checkpoint:

```bash
python training/train.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --weights_path path/to/checkpoint.pth
```

## Evaluation

```bash
python training/test.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --test_dataset "Celeb-DF-v2" \
  --weights_path path/to/checkpoint.pth
```

Use `--save_feat --feat_out_dir <dir>` to dump feature pickles for t-SNE or
other analysis.

## Git

This project is intended to be pushed to a new repository. It has no remote by
default after cleanup, so add your new remote explicitly:

```bash
git remote add origin <your-new-repo-url>
git push -u origin main
```
