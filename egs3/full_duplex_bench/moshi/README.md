# Full-Duplex-Bench v1.0 with Moshi

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025) measures turn-taking in full-duplex dialogue: whether a
model keeps quiet through the user's pauses and back-channels, takes the turn
when the user is done, and yields when interrupted. This recipe runs the
released Moshi on its v1.0 test sets and scores it with the benchmark's own
rules. See `egs3/TEMPLATE/moshi/README.md` for the system.

| Test set | Task | Samples | Audio |
| :-- | :-- | --: | --: |
| `candor_pause_handling` | Pause handling (Candor) | 216 | 42.6 min |
| `synthetic_pause_handling` | Pause handling (TTS) | 137 | 19.8 min |
| `icc_backchannel` | Backchannel (ICC) | 55 | 30.0 min |
| `candor_turn_taking` | Smooth turn-taking (Candor) | 119 | 27.8 min |
| `synthetic_user_interruption` | User interruption (TTS) | 200 | 93.3 min |

## Quick start

```bash
. ./path.sh

# 1) Download the benchmark (pinned revision) into downloads/
python run.py --stages create_dataset --inference_config conf/inference.yaml

# 2) One conversation per sample; Moshi's channel is listed in
#    exp/moshiko/inference/<test_set>/wav.scp
python run.py --stages infer --inference_config conf/inference.yaml

# 3) Score (GPU, see Requirements)
python run.py --stages measure \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

`infer` runs one GPU worker by default, at about 0.3 times real time on a
GH200 (about 70 min for the 214 min of user audio). To use every GPU of a
node, set in the inference config:

```yaml
parallel:
  env: local_gpu   # needs dask-cuda
  n_workers: 4     # number of GPUs
```

## Requirements

`measure` transcribes Moshi's channel with the benchmark's ASR,
`nvidia/parakeet-tdt-0.6b-v2`, through NeMo, and segments it with Silero VAD:

```bash
pip install "nemo_toolkit[asr]" silero-vad
```

NeMo pins `lightning<=2.4.0`, `hydra-core<=1.3.2` and `omegaconf<=2.3`, all
within ESPnet's own ranges. The results below used nemo_toolkit 3.0.0,
silero-vad 6.2.3 and lightning 2.4.0.

## Protocol

The metrics are a port of the benchmark's scripts (`src/metrics.py`). Laying
our outputs and their transcripts out as the benchmark expects and scoring
them with its own `eval_*.py` (commit 3e799c4) gives the numbers below to the
last printed digit, on all five test sets. What differs from the published
Moshi run:

- **Simulated call.** The published run streamed `input.wav` in real time to
  kyutai's `moshi.server` over a WebSocket, Opus-compressed. Here the infer
  stage feeds the same audio frame by frame to the same model, uncompressed
  and with no real-time constraint. The output layout is the client's: one
  silent frame, then Moshi's frames, as long as the input.
- **Sampling.** Moshi samples its tokens, so its behaviour varies from run to
  run. Each conversation here is seeded (`model.seed`) and computed with
  deterministic kernels, so the run is reproducible; the published run was
  one unseeded draw.
- **No GPT-4o score.** The relevance score of user interruption needs the
  OpenAI API and is not computed.

## Results

`exp_tag: moshiko`, `model.seed: 42424242` (the seed of `moshi.server`), on
GH200. The 95% intervals bootstrap the samples of this run; they do not cover
the variation between runs.

| Task | Score | Published Moshi | This recipe | 95% CI |
| :-- | :-- | --: | --: | :-- |
| Pause handling, synthetic | TOR ↓ | 0.985 | 0.993 | [0.978, 1.000] |
| Pause handling, Candor | TOR ↓ | 0.980 | 1.000 | [1.000, 1.000] |
| Backchannel | TOR ↓ | 1.000 | 0.400 | [0.273, 0.527] |
| Backchannel | Freq ↑ | 0.001 | 0.124 | |
| Backchannel | JSD ↓ | 0.957 | 0.724 | |
| Smooth turn-taking, Candor | TOR ↑ | 0.941 | 1.000 | [1.000, 1.000] |
| Smooth turn-taking, Candor | Latency (s) ↓ | 0.265 | 0.000 | [0.000, 0.000] |
| User interruption | TOR ↑ | 1.000 | 0.890 | [0.845, 0.930] |
| User interruption | Latency (s) ↓ | 0.257 | 1.164 | [0.910, 1.437] |

A frame took 24 to 26 ms on every test set, a third of the 80 ms budget.

### Reading the numbers

- **Moshi greets first.** In all 119 turn-taking conversations Moshi opens
  with a greeting ("Hey, how are you doing?") within the first second, while
  the user is still talking. The benchmark measures latency from Moshi's
  first word, so every turn-taking latency is clipped to 0, and the greeting
  alone counts as taking the turn in pause handling.
- **Backchannels.** On the ICC monologues Moshi mostly listens and says short
  "Yeah." backchannels, so it rarely takes the turn; the published run took
  it on every sample.
- **Interruptions.** The median latency after an interruption is 0.32 s,
  near the published mean; a tail of late replies raises the mean to 1.16 s.
- **This is Moshi, not the port.** kyutai's own `moshi==0.2.13`, fed the same
  frames, produces the same audio and text bit for bit, greeting included.
- **Not the codec.** Round-tripping the user audio through Opus (libopus,
  24 kbit/s VoIP, as kyutai's client and server do) changes little: on the
  backchannel set TOR 0.255, Freq 0.170, JSD 0.718; on turn-taking TOR 1.000,
  latency 0.000, and Moshi still greets in all 119 conversations.

The published run differs in ways this recipe does not reproduce (its server
implementation and version, real-time transport, an unseeded draw), and the
gap is larger than the sampling uncertainty above.
