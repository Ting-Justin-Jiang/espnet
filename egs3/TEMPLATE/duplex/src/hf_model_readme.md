---
tags:
- espnet
- ${system}
- ${corpus}
- full-duplex
language: ${lang}
license: ${license}
---

# ESPnet3 ${system} model

${description}

A full-duplex speech-text model fine-tuned with the ESPnet3 duplex system:
every 80 ms it reads one frame of the user's audio and writes one frame of
its own audio and inner-monologue text.

## Model

- Repository: `${hf_repo}`
- Recipe: `${recipe}`
- Corpus: `${corpus}`
- System: `${system}`
- Creator: `${creator}`
- Created: `${created_at}`
- Branch: `${git_branch}`
- Git: `${git_head}` (${git_dirty})
- Origin: ${git_origin}

${model_summary_section}
${model_detail_section}

## Inference

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("${hf_repo}")
```

`conf/inference.yaml` builds `espnet3.systems.duplex.inference.DuplexInference`
on the fine-tuned language model; Mimi and the text tokenizer are the
released ones, downloaded from the model block's `hf_repo`.

## Packaging

- Bundle: `${pack_name}`
- Exp dir: `${exp_dir}`
- Strategy: `${pack_strategy}`

${results_section}
${results_note}

## Training config

<details><summary>expand</summary>

```
${train_config}
```

</details>

### Citing ESPnet

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and
    Jiro Nishitoba and Yuya Unno and Nelson Yalta and Jahn Heymann and Matthew Wiesner
    and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456}
}
```
