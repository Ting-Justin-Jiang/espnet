"""otoSpeech full-duplex dataset module."""

from egs3.otospeech.duplex.dataset.builder import OtoSpeechBuilder as DatasetBuilder
from egs3.otospeech.duplex.dataset.dataset import OtoSpeechDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
