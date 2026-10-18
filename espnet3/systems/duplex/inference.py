"""Full-duplex Moshi inference: a user audio stream in, Moshi's stream out.

Built by a recipe's ``infer`` stage (``model._target_:
espnet3.systems.duplex.inference.MoshiInference``) and by the demo app. Both go
through :meth:`MoshiInference.step`, which exchanges one 80 ms frame: it
encodes the user's frame with Mimi, advances the language model by one frame
and decodes Moshi's frame. The model is causal, so a recorded user turn fed
frame by frame (:meth:`MoshiInference.__call__`) produces exactly what a live
conversation over the same audio would, without a microphone and without a
real-time constraint.

Output alignment: frame ``j`` of the returned audio is what Moshi emits after
user frame ``j``. Moshi's acoustic tokens are delayed by one frame, so the
first ``max_delay`` output frames are silence. This is the layout of the
Full-Duplex-Bench Moshi client, which writes one silent frame before the
first audio it receives.

Reproducibility: attention runs on the memory-efficient or math kernels. The
cuDNN attention kernel PyTorch picks by default on GH200 is not deterministic:
two runs over the same tokens differ after about 65 frames, and sampling then
makes the conversations diverge. With a fixed ``seed`` a conversation is
reproducible.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from espnet3.systems.duplex.src.cuda_graph import CUDAGraphed
from espnet3.systems.duplex.src.lm import LMGen, LMModel
from espnet3.systems.duplex.src.mimi import MimiModel
from espnet3.systems.duplex.src.streaming import start_streaming, stop_streaming

#: Text tokens Moshi emits while it has no word to say (padding, end of padding).
TEXT_PADDING_IDS = (0, 3)

#: Deterministic attention kernels (see the module docstring).
DETERMINISTIC_ATTENTION = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class MoshiInference:
    """Moshi with Mimi, run one frame at a time.

    Args:
        lm: Keyword arguments of
            :class:`~espnet3.systems.duplex.src.lm.LMModel`.
        mimi: Keyword arguments of
            :class:`~espnet3.systems.duplex.src.mimi.MimiModel`.
        hf_repo: Hugging Face repository holding the three files below, or
            ``None`` when they are local paths.
        revision: Repository revision; pin it for reproducible runs.
        lm_file: Language model weights (safetensors).
        mimi_file: Mimi weights (safetensors).
        tokenizer_file: SentencePiece model of the text stream.
        temp: Audio sampling temperature.
        temp_text: Text sampling temperature.
        top_k: Audio top-k.
        top_k_text: Text top-k.
        seed: Seed set at the start of every conversation, so that a
            conversation's output does not depend on the ones before it.
        cuda_graph: Whether to replay each frame from CUDA graphs, which real
            time needs on GPU. Must be ``False`` on CPU.
        device: Device to run on.
    """

    def __init__(
        self,
        lm: dict,
        mimi: dict,
        hf_repo: str | None,
        revision: str | None,
        lm_file: str,
        mimi_file: str,
        tokenizer_file: str,
        temp: float,
        temp_text: float,
        top_k: int,
        top_k_text: int,
        seed: int,
        cuda_graph: bool,
        device: str = "cuda",
    ) -> None:
        """Load the codec, the language model and the text tokenizer."""
        self.device = torch.device(device)
        if cuda_graph and self.device.type != "cuda":
            raise ValueError(f"cuda_graph needs a CUDA device, got {device}.")
        self.seed = seed
        self.cuda_graph = cuda_graph

        self.mimi = MimiModel(**mimi).to(self.device)
        self.mimi.load_state_dict(
            load_file(_resolve(hf_repo, revision, mimi_file), device=str(self.device))
        )
        self.mimi.eval()

        # The 7B model is created without memory and takes the loaded tensors,
        # in bfloat16 as released.
        with torch.device("meta"):
            self.lm = LMModel(**lm)
        state = load_file(_resolve(hf_repo, revision, lm_file), device=str(self.device))
        state = {key: value.to(torch.bfloat16) for key, value in state.items()}
        self.lm.load_state_dict(state, assign=True)
        self.lm.eval()

        self.lm_gen = LMGen(self.lm, temp, temp_text, top_k, top_k_text, cuda_graph)
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(_resolve(hf_repo, revision, tokenizer_file))
        )
        self.sample_rate = self.mimi.sample_rate
        self.frame_size = self.mimi.frame_size
        self._encode = None
        self._decode = None

    def start(self) -> None:
        """Open a conversation: fresh codec and language model streams, reseeded."""
        torch.manual_seed(self.seed)
        start_streaming(self.mimi, 1)
        self.lm_gen.start(1, self.device)
        # Graphs capture the stream state just created, so each conversation
        # gets its own.
        if self.cuda_graph:
            self._encode = CUDAGraphed(self.mimi.encode)
            self._decode = CUDAGraphed(self.mimi.decode)
        else:
            self._encode = self.mimi.encode
            self._decode = self.mimi.decode

    def stop(self) -> None:
        """Close the conversation and drop its graphs."""
        stop_streaming(self.mimi)
        self.lm_gen.stop()
        self._encode = None
        self._decode = None

    @property
    def is_streaming(self) -> bool:
        """Whether a conversation is open."""
        return self.lm_gen.is_streaming

    @torch.no_grad()
    def step(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """Exchange one frame of an open conversation.

        Args:
            frame: The user's next ``frame_size`` samples at ``sample_rate``.

        Returns:
            Moshi's next ``frame_size`` samples (silence before its first
            frame) and the text it emitted for that frame (empty when none).
        """
        if not self.is_streaming:
            raise RuntimeError("Call start() before step().")
        if frame.shape != (self.frame_size,):
            raise ValueError(f"Expected {self.frame_size} samples, got {frame.shape}.")
        user = torch.from_numpy(frame).to(self.device, torch.float32).view(1, 1, -1)
        with sdpa_kernel(DETERMINISTIC_ATTENTION):
            tokens = self.lm_gen.step(self._encode(user))
            if tokens is None:
                return np.zeros(self.frame_size, dtype=np.float32), ""
            pcm = self._decode(tokens[:, 1:])[0, 0].cpu().numpy()
        text_token = int(tokens[0, 0, 0])
        if text_token in TEXT_PADDING_IDS:
            return pcm, ""
        return pcm, self.tokenizer.id_to_piece(text_token).replace("\u2581", " ")

    def __call__(self, speech: np.ndarray) -> dict:
        """Run a whole conversation over a recorded user stream.

        Args:
            speech: The user's audio, mono, at ``sample_rate``.

        Returns:
            ``wav``: Moshi's audio, as long as ``speech`` and frame-aligned
            with it; ``text``: Moshi's inner-monologue text; ``step_ms``:
            median wall-clock time of a frame (real time requires under 80 ms).
            The median leaves out the first frames, which also capture the
            CUDA graphs.
        """
        num_frames = math.ceil(len(speech) / self.frame_size)
        padded = np.zeros(num_frames * self.frame_size, dtype=np.float32)
        padded[: len(speech)] = speech
        pcms = []
        pieces = []
        seconds = []
        self.start()
        for index in range(num_frames):
            begin = time.perf_counter()
            frame = padded[index * self.frame_size : (index + 1) * self.frame_size]
            pcm, piece = self.step(frame)
            seconds.append(time.perf_counter() - begin)
            pcms.append(pcm)
            pieces.append(piece)
        self.stop()
        return {
            "wav": np.concatenate(pcms)[: len(speech)],
            "text": "".join(pieces).strip(),
            "step_ms": 1000.0 * float(np.median(seconds)),
        }


def _resolve(hf_repo: str | None, revision: str | None, filename: str) -> Path:
    """Return ``filename`` downloaded from ``hf_repo``, or as a local path."""
    if hf_repo is None:
        return Path(filename)
    return Path(hf_hub_download(hf_repo, filename, revision=revision))
