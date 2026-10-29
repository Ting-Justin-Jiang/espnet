# ESPnet3 full-duplex recipe

A full-duplex speech-text model reads one 80 ms frame of the user's audio and
writes one frame of its own audio and inner-monologue text every step, so it
listens while it speaks. This system runs two released models of the same
architecture and fine-tunes them; each model is a block under `conf/model/`:

- `moshiko.yaml`: [Moshi](https://arxiv.org/abs/2410.00037)
  (`kyutai/moshiko-pytorch-bf16`).
- `personaplex.yaml`: [PersonaPlex](https://arxiv.org/abs/2602.06053)
  (`nvidia/personaplex-7b-v1`), Moshi fine-tuned to start every conversation
  from a voice prompt and a text role prompt. Its weights are gated: accept
  the NVIDIA Open Model License on the Hugging Face page first.

## Stages

A benchmark recipe scores released models (`DuplexEvaluationSystem`), as
egs2 benchmark recipes decode with `--skip_train true --download_model`: one
inference config per model, each with its own `exp_tag`.

```bash
python run.py --stages create_dataset infer measure \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
```

A fine-tuning recipe trains, then publishes the model and its demo
(`DuplexSystem`, see Fine-tuning):

```bash
python run.py --stages create_dataset train infer measure \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
python run.py --stages pack_model pack_demo --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml --demo_config conf/demo.yaml
```

`create_dataset` prepares the datasets of the training config, or the test
sets of the inference config in a benchmark recipe. `infer` runs one
conversation per test item. The inference config's `contract` checks what
goes in and out: the user's audio must be mono at 24 kHz (a test set
resamples it with its benchmark's resampler, so another rate is refused
rather than resampled again), and the outputs are the model's channel
(`<inference_dir>/<test>/wav.scp`), its inner monologue (`text.scp`) and the
median wall-clock time of a frame (`step_ms.scp`). `measure` scores them;
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

The model block is composed under `model:` by the inference and training
configs (`defaults: [model: moshiko]`). Another released Moshi, such as
`kyutai/moshika-pytorch-bf16`, is a copy of `moshiko.yaml` with its repository
and revision.

## Fine-tuning

`conf/training.yaml` fine-tunes the released model of `pretrained` in full,
with the settings of kyutai's
[moshi-finetune](https://github.com/kyutai-labs/moshi-finetune): 100 s
windows, 16 windows a step, AdamW at 2e-6 with a one-cycle schedule and 5 %
warm-up, gradient norm clipped at 1. `src/training.py` holds the two pieces a
recipe plugs in:

- **The data.** A training item is a window of a two-channel conversation
  at 24 kHz, channel 0 being the speaker the model learns to be, with the
  timestamped words of channel 0. The dataset preprocessor `TextStream`
  writes them into the text stream as moshi-finetune's interleaver does.
- **The model.** `DuplexLM` encodes both channels with the frozen Mimi in
  float32, as at inference, and trains the language model on every stream:
  text cross-entropy with the padding tokens weighted 0.5, audio
  cross-entropy with the semantic codebook weighted 100.

The training forward gives, for every stream and frame, the logits that
frame-by-frame generation computes from the same tokens, and no prediction
depends on its own target (`test_lm.py`).

The 7B model trains with FSDP. Parameters, gradients and AdamW states are
sharded; every transformer layer recomputes its activations; parameters and
buffers stay float32 (bfloat16 buffers would round Mimi's codebooks) while
autocast computes in bfloat16; `FSDPClipGradNorm` clips the gradient norm
over all shards, which Lightning's own clipping does not do under FSDP. On
3 GH200 (96 GB each), four 100 s windows a GPU took about 73 GB. The
checkpoint with the lowest validation loss, `valid.loss.ave_1best.pth`, is
read by inference as `model.lm_checkpoint`.

PersonaPlex's voice and role prompts are not part of the training items:
fine-tuning `pretrained: model/personaplex` would train it without them.

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

`pack_model` bundles the inference config with the experiment directory
(`exp/<exp_tag>/model_pack`); `pack_demo` writes a Gradio app (`src/app.py`)
with `demo.yaml` into `exp/<exp_tag>/demo/`, pointing at that bundle. The app
loads the bundle through `load_demo_session`, that is through
`InferenceModel`, as every ESPnet3 demo does, and drives its
`DuplexInference` frame by frame: the browser streams the microphone to the
app, which resamples each chunk to 24 kHz with a streaming resampler, feeds
every complete frame to `DuplexInference.step` and streams the model's frames
back. It is the same `step` the `infer` stage uses. One conversation runs at a
time; a PersonaPlex demo opens each with the demo config's `role_prompt`.

On a cluster, run the app on a GPU node and reach it through SSH:

```bash
# on the GPU node (inside an interactive job), after pack_model
python run.py --stages pack_demo --training_config conf/training.yaml \
    --demo_config conf/demo.yaml
python exp/<exp_tag>/demo/app.py --demo-dir exp/<exp_tag>/demo --server-port 7860

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

`upload_demo` publishes the packed app as a Hugging Face Space; the model
needs a GPU Space to run in real time.
