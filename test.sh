python training/test.py \
  --detector_path training/config/detector/BiasLoraAsy.yaml \
  --test_dataset  "uniface_ff" "blendface_ff" "e4s_ff" "facedancer_ff" "fsgan_ff" "inswap_ff" "simswap_ff" \
  --weights_path /kaggle/working/DeepfakeBench/training/pretrained/ckpt_best.pth

#--test_dataset  "UADFV" "Celeb-DF-v2"  "DFDCP" "DeepFakeDetection" "DFDC" "uniface_ff" "blendface_ff" "e4s_ff" "facedancer_ff" "fsgan_ff" "inswap_ff" "simswap_ff" \
