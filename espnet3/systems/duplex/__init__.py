"""Moshi, a full-duplex speech-text foundation model, for ESPnet3.

Moshi listens and speaks at the same time: at every 80 ms frame it reads the
user's audio and writes its own audio and inner-monologue text. The package
holds an inference-only port of the released PyTorch model (``src/``: the Mimi
codec, the language model and its frame-by-frame generator), the duplex
inference engine (``inference``) and the staged pipeline a recipe under
``egs3/<corpus_name>/duplex/`` runs (``system``).

The model is described in:

Alexandre Defossez, Laurent Mazare, Manu Orsini, Amelie Royer, Patrick Perez,
Herve Jegou, Edouard Grave, Neil Zeghidour.
"Moshi: a speech-text foundation model for real-time dialogue"
https://arxiv.org/abs/2410.00037

If you use this model in your research, please cite the paper above.
"""
