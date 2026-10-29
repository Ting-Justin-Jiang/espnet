# Full-Duplex-Bench v1.0 with Moshi and PersonaPlex

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025) measures turn-taking in full-duplex dialogue: whether a
model keeps quiet through the user's pauses and back-channels, takes the turn
when the user is done, and yields when interrupted. This recipe runs the
released Moshi and PersonaPlex on its v1.0 test sets and scores them with
the benchmark's own rules. See `egs3/TEMPLATE/duplex/README.md` for the
system.

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

# PersonaPlex: the same, with its config (accept its license on the Hub first)
python run.py --stages infer measure \
    --inference_config conf/tuning/inference_personaplex.yaml \
    --metrics_config conf/metrics.yaml
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

`measure` transcribes the model's channel with the benchmark's ASR,
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
them with its own `eval_*.py` (commit 3e799c4) gives the same numbers to the
last printed digit, on all five test sets. The relevance score of user
interruption needs the OpenAI API and is not computed.

Each model is reported three ways, all scored by this recipe's `measure`:

- **Paper**: the published row (Moshi from the Full-Duplex-Bench paper,
  PersonaPlex from its own paper; both are on the benchmark's leaderboard).
- **Official code**: the evaluation path the benchmark prescribes, run on the
  same GH200 node with the released checkpoint. For Moshi, kyutai's
  `moshi.server` (`moshi==0.2.13`) driven by the benchmark's own client
  (`v1_v1.5/model_inference/moshi/inference.py`), which streams `input.wav`
  as Opus over a WebSocket at real-time pace and records the reply as it
  arrives. 2 of the server's 160,000 frames took over 80 ms (at most
  90 ms); the client records frames as they arrive, so they do not shift the
  recording. The client is written against sphn 0.1: with sphn 0.2, which
  the server needs, it silently records silence, so it runs with sphn
  0.1.12. For PersonaPlex, NVIDIA's `moshi/offline.py` (the benchmark points
  to it), voice `NATF2.pt`, the benchmark's role prompts.
- **ESPnet3**: this recipe, three seeds (`model.seed`).

Seeds matter more than the number of samples suggests. A conversation is
reseeded when it starts, so every conversation of a run draws the same
random numbers at the same frames: whether the model speaks up in its first
second, for one, is then decided alike across a whole test set. Two
PersonaPlex runs differ by up to 0.18 in pause-handling TOR, about four
times what independent conversations would give. The tables therefore give
the mean and standard deviation over seeds, against which the official
code's single run is read.

## Results

Lower is better for ↓, higher for ↑. Latency is in seconds.

### Moshi

| Task | Score | Paper | Official pipeline, 1 run | ESPnet3, 3 seeds |
| :-- | :-- | --: | --: | --: |
| Pause, synthetic | TOR ↓ | 0.985 | 1.000 | 0.998 ± 0.004 |
| Pause, Candor | TOR ↓ | 0.980 | 1.000 | 1.000 ± 0.000 |
| Backchannel | TOR ↓ | 1.000 | 0.291 | 0.424 ± 0.021 |
| Backchannel | Freq ↑ | 0.001 | 0.172 | 0.144 ± 0.021 |
| Backchannel | JSD ↓ | 0.957 | 0.693 | 0.712 ± 0.011 |
| Turn-taking | TOR ↑ | 0.941 | 1.000 | 1.000 ± 0.000 |
| Turn-taking | Latency (s) ↓ | 0.265 | 0.000 | 0.000 ± 0.000 |
| Interruption | TOR ↑ | 1.000 | 0.890 | 0.843 ± 0.060 |
| Interruption | Latency (s) ↓ | 0.257 | 0.883 | 1.104 ± 0.150 |

### PersonaPlex

| Task | Score | Paper | NVIDIA offline.py, 1 run | ESPnet3, 3 seeds |
| :-- | :-- | --: | --: | --: |
| Pause, synthetic | TOR ↓ | 0.584 | 0.212 | 0.263 ± 0.070 |
| Pause, Candor | TOR ↓ | 0.662 | 0.343 | 0.401 ± 0.098 |
| Backchannel | TOR ↓ | 0.327 | 0.309 | 0.188 ± 0.042 |
| Backchannel | Freq ↑ | 0.025 | 0.103 | 0.113 ± 0.005 |
| Backchannel | JSD ↓ | 0.649 | 0.767 | 0.747 ± 0.003 |
| Turn-taking | TOR ↑ | 0.992 | 0.941 | 0.952 ± 0.013 |
| Turn-taking | Latency (s) ↓ | 0.070 | 0.428 | 0.357 ± 0.146 |
| Interruption | TOR ↑ | 1.000 | 0.930 | 0.938 ± 0.013 |
| Interruption | Latency (s) ↓ | 0.400 | 0.182 | 0.198 ± 0.011 |

A median frame took 23.5 ms for Moshi and 29.7 ms for PersonaPlex (its
depformer samples 16 codebooks, Moshi's 8), within the 80 ms of a frame.

### Reading the numbers

- **ESPnet3 runs the official models.** The port is bit-exact with kyutai's
  `moshi==0.2.13` on Moshi, eager and with CUDA graphs, and emits the same
  text and audio tokens as NVIDIA's `offline.py` on PersonaPlex under greedy
  sampling. With sampling, conversation by conversation, the official runs
  do not differ from ESPnet3's seed-42424242 run in turn-over rate on any
  task (Fisher's exact test, p ≥ 0.32), nor in latency (Mann-Whitney U,
  p ≥ 0.12, but for PersonaPlex's interruption latency, p = 0.03 with equal
  medians of 0.24 s).
- **Moshi greets first, with its own server too.** In every pause-handling,
  backchannel and turn-taking conversation Moshi opens with a greeting
  ("Good day. How is it going?") within its first two seconds, while the
  user is still talking, through kyutai's server as through ESPnet3. The
  greeting counts as taking the turn in pause handling, and turn-taking
  latency is measured from the model's first word, so it is 0. The published
  Moshi row (latency 0.265 s, backchannel TOR 1.000) is not what the
  released model does through the official pipeline; the paper does not
  give the version or settings it ran.
- **Opus lowers Moshi's backchannel TOR.** The official client sends the
  user's audio as Opus. Its backchannel TOR, 0.291, is below ESPnet3's; an
  Opus round trip of the user audio before ESPnet3 gives 0.255.
- **PersonaPlex takes the turn less than published.** Its pause-handling
  TOR is about half the published value, with NVIDIA's own code as with
  ESPnet3, and on a second voice too (`NATM1.pt`: synthetic pauses 0.161,
  turn-taking TOR 0.950). Most of these turn-overs start before the user's
  first pause: the model speaks up early in some conversations, and how
  often depends on the seed. The paper does not give its voice prompt or
  sampling draw; the released checkpoint and script do not reproduce its
  pause numbers.
- **Open: PersonaPlex's backchannel TOR.** NVIDIA's code took the turn in
  0.309 and 0.364 of the backchannel conversations (two seeds), ESPnet3 in
  0.164 to 0.236. The ICC inputs are 48 kHz; resampling them with sphn, as
  NVIDIA's script does, instead of torchaudio, as the benchmark's client
  does, gives ESPnet3 0.255. With this run-to-run spread the remaining
  difference is not established either way.
