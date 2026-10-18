"""Full-duplex speech-text models for ESPnet3: Moshi and PersonaPlex.

A full-duplex model listens and speaks at the same time: at every 80 ms frame
it reads the user's audio and writes its own audio and inner-monologue text.
The package holds a port of the released PyTorch models (``src/``: the Mimi
codec, the language model, its frame-by-frame generators and its
fine-tuning), the duplex inference engine (``inference``) and the staged
pipelines recipes under ``egs3/<corpus_name>/duplex/`` run (``system``).

The models are described in:

Alexandre Defossez, Laurent Mazare, Manu Orsini, Amelie Royer, Patrick Perez,
Herve Jegou, Edouard Grave, Neil Zeghidour.
"Moshi: a speech-text foundation model for real-time dialogue"
https://arxiv.org/abs/2410.00037

Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby,
Sungwon Kim, Jaehyeon Kim, Bryan Catanzaro.
"PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech
Models"
https://arxiv.org/abs/2602.06053

If you use these models in your research, please cite the papers above.
"""
