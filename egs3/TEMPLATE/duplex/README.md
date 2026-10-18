# ESPnet3 full-duplex recipe

A full-duplex speech-text model reads one 80 ms frame of the user's audio and
writes one frame of its own audio and inner-monologue text every step, so it
listens while it speaks. This system runs two released models of the same
architecture, each described by a block under `conf/model/`:

- `moshiko.yaml`: [Moshi](https://arxiv.org/abs/2410.00037)
  (`kyutai/moshiko-pytorch-bf16`).
- `personaplex.yaml`: [PersonaPlex](https://arxiv.org/abs/2602.06053)
  (`nvidia/personaplex-7b-v1`), Moshi fine-tuned to start every conversation
  from a voice prompt and a text role prompt. Its weights are gated: accept
  the NVIDIA Open Model License on the Hugging Face page first.

## Stages

```bash
python run.py --stages create_dataset infer --inference_config conf/inference.yaml
python run.py --stages measure \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
python run.py --stages pack_demo --demo_config conf/demo.yaml
```

`create_dataset` prepares the test sets of the inference config. `infer` runs
one conversation per test item and writes the model's channel
(`<inference_dir>/<test>/wav/*.wav`), its inner monologue (`text.scp`) and the
mean wall-clock time of a frame (`step_ms.scp`). `measure` scores them;
`StepTime` reports the frame time against the 80 ms real-time budget.

## The model

`espnet3/systems/duplex/src` ports the PyTorch inference code of
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) (`moshi==0.2.13`,
MIT). The `moshi` package itself cannot be a dependency: it requires
`torch<2.10`, and ESPnet supports only `torch>=2.11`. Parameter names follow
the released checkpoints, which load with `strict=True`.

The port was checked against `moshi==0.2.13` on a GH200: the same Mimi codes
and audio, and the same 20 s duplex conversation (audio and text, bit for
bit), both eager and with CUDA graphs.

`src/personaplex.py` ports PersonaPlex's generator from
[NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) (MIT). The network
is Moshi's, except that the depformer also predicts the user's codebooks
(`dep_q: 16`); the generator feeds the user's real codes in their place, and
runs the voice and role prompts before the user's first frame. With greedy
sampling, the port emitted the same text and audio tokens as NVIDIA's offline
script on two Full-Duplex-Bench inputs (136 and 344 frames). Seeded sampling
cannot be compared: the reference draws random numbers while it builds and
warms up its models.

Two choices matter for experiments:

- **Determinism.** On GH200, PyTorch picks the cuDNN attention kernel, which
  is not deterministic: two runs over the same tokens differ after about 65
  frames, and sampling then makes the conversations diverge. `DuplexInference`
  restricts attention to the memory-efficient and math kernels, which are, and
  reseeds at the start of every conversation (`model.seed`). A conversation is
  then reproducible and independent of the others.
- **Real time.** A frame launches a few thousand small kernels. Eagerly that
  takes about 105 ms per frame, more than the frame lasts. With
  `model.cuda_graph: true` each frame replays CUDA graphs and takes about
  25 ms. The output is identical either way; only live serving needs the
  graphs.

The model block is composed under `model:` by the inference and demo configs
(`defaults: [model: moshiko]`). Another released Moshi, such as
`kyutai/moshika-pytorch-bf16`, is a copy of `moshiko.yaml` with its repository
and revision.

## Serving without a microphone

The `infer` stage is the serving loop with recorded audio in place of the
microphone. `DuplexInference.__call__` feeds the user's audio to
`DuplexInference.step` one frame at a time, exactly as a live session does,
and collects Moshi's frames. Moshi is causal, so frame `j` of the output is
what Moshi would have said live after hearing frame `j` of the input: the
two channels are aligned, as a two-channel recording of the call would be.
The first frame is silent because Moshi's acoustic tokens are delayed by one
frame; this is also the layout of the Full-Duplex-Bench Moshi client.
PersonaPlex's frame `j` is what it emits after user frame `j`, one frame
later in the conversation, as in its offline script; its role prompt is an
argument of the conversation (`DuplexInference.start`).

The simulation has no real-time constraint, so its output does not depend on
the GPU. Whether a GPU could keep up live is what `step_ms` measures.
`egs3/full_duplex_bench/duplex` uses this to evaluate Moshi on
Full-Duplex-Bench.

## Talking to it with a microphone

`pack_demo` writes a Gradio app (`src/app.py`) with `demo.yaml` into
`exp/<exp_tag>/demo/`.
The browser streams the microphone to the app; the app resamples each chunk
to 24 kHz with a streaming resampler, feeds every complete frame to
`DuplexInference.step` and streams Moshi's frames back. It is the same `step`
the `infer` stage uses. One conversation runs at a time.

On a cluster, run the app on a GPU node and reach it through SSH:

```bash
# on the GPU node (inside an interactive job)
python run.py --stages pack_demo --demo_config conf/demo.yaml
python exp/moshiko/demo/app.py --demo-dir exp/moshiko/demo --server-port 7860

# on your laptop: forward the port through the login node to the GPU node
ssh -N -L 7860:localhost:7860 -J <user>@<login-node> <user>@<gpu-node>
# then open http://localhost:7860 and press record (use headphones)
```

Driven by a recording in 80 ms microphone chunks at real-time pace, the
packed app answered every chunk in at most 24 ms on a GH200 (median 23.7 ms),
with two warm-up frames at the start of the conversation.

Browsers give the microphone only to HTTPS pages and to `localhost`, which the
tunnel provides. Gradio streams over the same HTTP connection, so one TCP
tunnel is enough; a WebRTC client would need UDP, which SSH does not forward.
Each microphone chunk is one frame (`stream_every=0.08`); the network and the
browser add latency on top of Moshi's own.

`upload_demo` publishes the packed app as a Hugging Face Space; Moshi needs a
GPU Space to run in real time.
