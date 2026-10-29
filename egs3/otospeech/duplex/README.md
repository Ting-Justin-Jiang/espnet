# otoSpeech full-duplex fine-tuning

Fine-tunes the released Moshi on
[otoSpeech-full-duplex-processed-141h](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h)
(CC BY 4.0, gated: request access with the Hugging Face account you log in
with) and evaluates it on Full-Duplex-Bench v1.0. See
`egs3/TEMPLATE/duplex/README.md` for the system.

## Quick start

```bash
. ./path.sh

# 1) Download the shards, resample each session to 24 kHz and transcribe
#    each channel with word timestamps (GPU, NeMo)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Fine-tune with FSDP (4 GPUs)
python run.py --stages train --training_config conf/training.yaml

# 3) Evaluate on Full-Duplex-Bench (run create_dataset in
#    egs3/full_duplex_bench/duplex once, to download it)
python run.py --stages infer measure --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 4) Pack the model and its microphone demo
python run.py --stages pack_model pack_demo --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml --demo_config conf/demo.yaml
```

## Data

A session is one two-person call, one speaker per channel. Both speakers are
people, so each takes the model's side in turn: a 100 s window of a session
gives two training items, one with each speaker on channel 0 (the stream the
model learns to produce) and the other as the user. The words of the model's
channel come from `nvidia/parakeet-tdt-0.6b-v2`, the ASR Full-Duplex-Bench
scores with. The corpus has no official split; the last 6 sessions (by id)
validate.

The data format follows the dataset card (WebDataset shards of a
two-channel FLAC and a JSON per session) and has not been checked against
the shards yet; `create_dataset` refuses a session that is not two channels
mapped to speakers 0 and 1.
