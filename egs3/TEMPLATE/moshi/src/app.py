"""Full-duplex Moshi demo: talk with Moshi through the browser microphone.

The browser streams the microphone to this app over the Gradio connection.
Each chunk is resampled to 24 kHz and cut into 80 ms frames; every frame goes
through ``MoshiInference.step``, which returns Moshi's next 80 ms, streamed
back to the browser. Moshi keeps listening while it speaks, so the user can
interrupt it or back-channel. One conversation runs at a time: the model's
stream state lives on the GPU.

Browsers allow the microphone only on HTTPS or on ``localhost``; on a cluster
node, reach the app through an SSH tunnel (see the system README).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import numpy as np
import soxr
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS, UIAsset
from espnet3.publication.demo.session import DemoSession
from espnet3.systems.moshi.inference import MoshiInference
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


class MicrophoneStreamUI(UIAsset):
    """Microphone input streamed to the app while recording."""

    def build_input(self, spec):
        """Build a streaming microphone component."""
        return self.gradio_module.Audio(
            sources=["microphone"], streaming=True, type="numpy", label=spec["label"]
        )


class AudioStreamUI(UIAsset):
    """Audio output played as its chunks arrive."""

    def build_output(self, spec):
        """Build a streaming, autoplaying audio component."""
        return self.gradio_module.Audio(
            streaming=True, autoplay=True, label=spec["label"]
        )


class DuplexConversation:
    """One live conversation between the browser microphone and Moshi.

    Args:
        model: The Moshi engine, shared by the conversations one at a time.
    """

    def __init__(self, model: MoshiInference) -> None:
        """Keep the model; the conversation starts with :meth:`start`."""
        self.model = model
        self.resampler: soxr.ResampleStream | None = None
        self.input_rate: int | None = None
        self.pending = np.zeros(0, dtype=np.float32)
        self.text = ""

    def start(self) -> None:
        """Begin a conversation, closing one the browser left open."""
        if self.model.is_streaming:
            self.model.stop()
        self.model.start()
        self.resampler = None
        self.input_rate = None
        self.pending = np.zeros(0, dtype=np.float32)
        self.text = ""

    def stop(self) -> None:
        """End the conversation."""
        if self.model.is_streaming:
            self.model.stop()

    def __call__(self, *, speech) -> dict:
        """Feed one microphone chunk and return Moshi's reply to it.

        Args:
            speech: ``(sample_rate, samples)`` from the microphone, int16 or
                float, mono or multi-channel.

        Returns:
            ``wav``: ``(24000, samples)`` Moshi produced for the complete
            frames received; ``text``: the inner monologue so far.
        """
        rate, samples = speech
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)
        if self.resampler is None:
            self.input_rate = rate
            self.resampler = soxr.ResampleStream(
                rate, self.model.sample_rate, 1, dtype="float32"
            )
        if rate != self.input_rate:
            raise ValueError(
                f"Microphone rate changed from {self.input_rate} to {rate}."
            )
        resampled = self.resampler.resample_chunk(samples)
        self.pending = np.concatenate([self.pending, resampled])

        frame_size = self.model.frame_size
        replies = []
        while len(self.pending) >= frame_size:
            pcm, piece = self.model.step(self.pending[:frame_size])
            self.pending = self.pending[frame_size:]
            replies.append(pcm)
            self.text += piece
        if replies:
            reply = np.concatenate(replies)
        else:
            reply = np.zeros(0, dtype=np.float32)
        return {"wav": (self.model.sample_rate, reply), "text": self.text.strip()}


def build_model(model_config) -> MoshiInference:
    """Build the Moshi engine the demo config's ``model`` block describes."""
    arguments = OmegaConf.to_container(model_config, resolve=True)
    # Recorded for pack_demo; the weights come from the block's hf_repo.
    del arguments["dir_or_tag"]
    return instantiate(arguments)


def build_demo(demo_dir: Path, demo_config_path: Path | None = None):
    """Build the Gradio app of one packed demo."""
    demo_dir = Path(demo_dir).resolve()
    if demo_config_path is None:
        demo_config_path = demo_dir / "demo.yaml"
    config = load_config_with_defaults(str(demo_config_path))
    conversation = DuplexConversation(build_model(config.model))

    registry = DEFAULT_UI_ASSETS.clone()
    registry.register("microphone_stream", MicrophoneStreamUI)
    registry.register("audio_stream", AudioStreamUI)
    session = DemoSession(demo_dir, config, conversation, registry)
    respond = session.create_inference_fn(session.input_specs, session.output_specs)

    with gr.Blocks(title=session.title or "Moshi") as app:
        if session.title:
            gr.Markdown(f"# {session.title}")
        with gr.Row():
            microphone = session.build_input_component(session.input_specs[0])
            outputs = [session.build_output_component(s) for s in session.output_specs]
        if session.description:
            gr.Markdown(session.description)
        microphone.start_recording(conversation.start)
        # One chunk per frame keeps the reply's added delay at one frame.
        microphone.stream(
            respond,
            inputs=[microphone],
            outputs=outputs,
            stream_every=0.08,
            concurrency_limit=1,
        )
        microphone.stop_recording(conversation.stop)
    return app


def main() -> None:
    """Launch the packed demo."""
    parser = argparse.ArgumentParser(description="Launch the Moshi duplex demo.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Packed demo directory. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--demo-config",
        type=Path,
        default=None,
        help="Packed demo config. Defaults to <demo-dir>/demo.yaml.",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Address to listen on; 0.0.0.0 makes it reachable from other hosts.",
    )
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    configure_logging(log_dir=args.demo_dir, filename="demo.log")
    app = build_demo(args.demo_dir, args.demo_config)
    app.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
