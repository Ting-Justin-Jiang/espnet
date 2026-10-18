# ESPnet3 Moshi recipe

[Moshi](https://arxiv.org/abs/2410.00037) is a full-duplex speech-text model:
every 80 ms it reads one frame of the user's audio and writes one frame of its
own audio and inner-monologue text, so it can listen while it speaks. This
system runs the released model (`kyutai/moshiko-pytorch-bf16`); it has no
training stage yet.

## Stages

```bash
python run.py --stages create_dataset infer --inference_config conf/inference.yaml
python run.py --stages measure \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml
python run.py --stages pack_demo --demo_config conf/demo.yaml
```

`create_dataset` prepares the test sets of the inference config. `infer` runs
one conversation per test item and writes Moshi's channel
(`<inference_dir>/<test>/wav/*.wav`), its inner monologue (`text.scp`) and the
mean wall-clock time of a frame (`step_ms.scp`). `measure` scores them;
`StepTime` reports the frame time against the 80 ms real-time budget.

## The model

`espnet3/systems/moshi/src` ports the PyTorch inference code of
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) (`moshi==0.2.13`,
MIT). The `moshi` package itself cannot be a dependency: it requires
`torch<2.10`, and ESPnet supports only `torch>=2.11`. Parameter names follow
the released checkpoints, which load with `strict=True`.

The port was checked against `moshi==0.2.13` on a GH200: the same Mimi codes
and audio, and the same 20 s duplex conversation (audio and text, bit for
bit), both eager and with CUDA graphs.

Two choices matter for experiments:

- **Determinism.** On GH200, PyTorch picks the cuDNN attention kernel, which
  is not deterministic: two runs over the same tokens differ after about 65
  frames, and sampling then makes the conversations diverge. `MoshiInference`
  restricts attention to the memory-efficient and math kernels, which are, and
  reseeds at the start of every conversation (`model.seed`). A conversation is
  then reproducible and independent of the others.
- **Real time.** A frame launches a few thousand small kernels. Eagerly that
  takes about 105 ms per frame, more than the frame lasts. With
  `model.cuda_graph: true` each frame replays CUDA graphs and takes about
  25 ms. The output is identical either way; only live serving needs the
  graphs.

The model block is `conf/model/moshiko.yaml`, composed under `model:` by the
inference and demo configs. Another released Moshi, such as
`kyutai/moshika-pytorch-bf16`, is a copy of that file with its repository and
revision.

## Serving without a microphone

The `infer` stage is the serving loop with recorded audio in place of the
microphone. `MoshiInference.__call__` feeds the user's audio to
`MoshiInference.step` one frame at a time, exactly as a live session does,
and collects Moshi's frames. Moshi is causal, so frame `j` of the output is
what Moshi would have said live after hearing frame `j` of the input: the
two channels are aligned, as a two-channel recording of the call would be.
The first frame is silent because Moshi's acoustic tokens are delayed by one
frame; this is also the layout of the Full-Duplex-Bench Moshi client.

The simulation has no real-time constraint, so its output does not depend on
the GPU. Whether a GPU could keep up live is what `step_ms` measures.

## Talking to it with a microphone

`pack_demo` writes a Gradio app (`src/app.py`) with `demo.yaml` into
`exp/<exp_tag>/demo/`.
The browser streams the microphone to the app; the app resamples each chunk
to 24 kHz with a streaming resampler, feeds every complete frame to
`MoshiInference.step` and streams Moshi's frames back. It is the same `step`
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

Browsers give the microphone only to HTTPS pages and to `localhost`, which the
tunnel provides. Gradio streams over the same HTTP connection, so one TCP
tunnel is enough; a WebRTC client would need UDP, which SSH does not forward.
Each microphone chunk is one frame (`stream_every=0.08`); the network and the
browser add latency on top of Moshi's own.

`upload_demo` publishes the packed app as a Hugging Face Space; Moshi needs a
GPU Space to run in real time.
