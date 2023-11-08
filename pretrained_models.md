# Available pretrained models

## Download links

| Subdirectory | Model (link to README) | Model shortname | Link | Main results|
|---|---|---|---|---|
| `retrieval` | [text-to-pose retrieval model](./src/text2pose/retrieval/README.md) | `ret_distilbert_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/ret_distilbert_dataPSA2ftPSH2.zip) | **mRecall** = 47.92 <br> **R@1 Precision (GT)** = 84.76 |
| `retrieval_modifier` | [pose-pair-to-instruction retrieval model](./src/text2pose/retrieval_modifier/README.md) | `modret_distilbert_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/modret_distilbert_dataPFAftPFH.zip) | **mRecall** = 30.00 <br> **R@1 Precision (GT)** = 68.04 |
| `generate` | [text-conditioned pose generation model](./src/text2pose/generative/README.md) | `gen_distilbert_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/gen_distilbert_dataPSA2ftPSH2.zip) | **ELBO jts/vert/rot** = 1.44 / 1.82 / 0.90 |
| `generate_B` | [text-guided pose editing model](./src/text2pose/generative_B/README.md) | `b_gen_distilbert_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/b_gen_distilbert_dataPFAftPFH.zip) | **ELBO jts/vert/rot** = 1.43 / 1.90 / 1.00 |
| `generate_caption` | [pose description generation model](./src/text2pose/generative_caption/README.md) | `capgen_CAtransfPSA2H2_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip) | **R@1 Precision** = 89.38 <br> **MPJE_30** = 202 <br> **ROUGE-L** = 33.95 | 
| `generate_modifier` | [pose-based correctional text generation model](./src/text2pose/generative_modifier/README.md) | `modgen_CAtransfPFAHPP_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/modgen_CAtransfPFAHPP_dataPFAftPFH.zip) |  **R@1 Precision** = 78.85 <br> **MPJE_30** = 186 <br> **ROUGE-L** = 33.53 | 

Unzip the archives and place the content of the resulting directory in ***GENERAL_EXP_OUTPUT_DIR***.

## References in `shortname_2_model_path.txt`

References should be given the following format:

```
<model shortname><4 spaces><path to the model>
```

Thus, for the above-mentioned models (simply replace `<GENERAL_EXP_OUTPUT_DIR>` by its proper value):
```text
ret_distilbert_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/ret_distilbert_dataPSA2ftPSH2/checkpoint_best.pth
modret_distilbert_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/modret_distilbert_dataPFAftPFH/checkpoint_best.pth
gen_distilbert_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/gen_distilbert_dataPSA2ftPSH2/checkpoint_best.pth
b_gen_distilbert_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/b_gen_distilbert_dataPFAftPFH/checkpoint_best.pth
capgen_CAtransfPSA2H2_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/capgen_CAtransfPSA2H2_dataPSA2ftPSH2/checkpoint_best.pth
modgen_CAtransfPFAHPP_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/modgen_CAtransfPFAHPP_dataPFAftPFH/checkpoint_best.pth
```