# Available trained models

## Download links

| Subdirectory | Model (link to README) | Model shortname | Link | Main results|
|---|---|---|---|---|
| `retrieval` | [text-to-pose retrieval model](./src/text2pose/retrieval/README.md) | `ret_distilbert_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/ret_distilbert_dataPSA2ftPSH2.zip) | **mRecall** = 47.92 <br> **R@1 Precision (GT)** = 84.76 |
| `retrieval_modifier` | [pose-pair-to-instruction retrieval model](./src/text2pose/retrieval_modifier/README.md) | `modret_distilbert_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/modret_distilbert_dataPFAftPFH.zip) | **mRecall** = 30.00 <br> **R@1 Precision (GT)** = 68.04 |
| `generative` | [text-conditioned pose generation model](./src/text2pose/generative/README.md) | `gen_distilbert_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/gen_distilbert_dataPSA2ftPSH2.zip) | **ELBO jts/vert/rot** = 1.44 / 1.82 / 0.90 |
| `generative_B` | [text-guided pose editing model](./src/text2pose/generative_B/README.md) | `b_gen_distilbert_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/b_gen_distilbert_dataPFAftPFH.zip) | **ELBO jts/vert/rot** = 1.43 / 1.90 / 1.00 |
| `generative_caption` | [pose description generation model](./src/text2pose/generative_caption/README.md) | `capgen_CAtransfPSA2H2_dataPSA2ftPSH2` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip) | **R@1 Precision** = 89.38 <br> **MPJE_30** = 202 <br> **ROUGE-L** = 33.95 | 
| `generative_modifier` | [pose-based correctional text generation model](./src/text2pose/generative_modifier/README.md) | `modgen_CAtransfPFAHPP_dataPFAftPFH` | [download](https://download.europe.naverlabs.com/ComputerVision/PoseFix/modgen_CAtransfPFAHPP_dataPFAftPFH.zip) |  **R@1 Precision** = 78.85 <br> **MPJE_30** = 186 <br> **ROUGE-L** = 33.53 | 

Unzip the archives and place the content of the resulting directories in ***GENERAL_EXP_OUTPUT_DIR***.

**Note:** these models are the result of a two-stage training, involving a pretraining stage on automatic texts, and a finetuning stage on human-written annotations.

<details>
<summary>Bash script to download & unzip everything all at once.</summary>

```bash
cd "<GENERAL_EXP_OUTPUT_DIR>" # TODO replace!

arr=(
    ret_distilbert_dataPSA2ftPSH2
    modret_distilbert_dataPFAftPFH
    gen_distilbert_dataPSA2ftPSH2
    b_gen_distilbert_dataPFAftPFH
    capgen_CAtransfPSA2H2_dataPSA2ftPSH2
    modgen_CAtransfPFAHPP_dataPFAftPFH
)

for a in "${arr[@]}"; do
    echo "Download and extract $a"
    wget "https://download.europe.naverlabs.com/ComputerVision/PoseFix/${a}.zip"
    unzip "${a}.zip"
    rm "${a}.zip"
done
```

</details>

<details>
<summary>Differences in results with the papers.</summary>

* *Text-to-pose retrieval*: providing an improved model, pretrained on new automatic captions, and with a symmetric constrastive loss (vs. uni-directional contrastive loss in the paper)
* *Instruction-to-pair retrieval*: providing an improved model trained with symmetric contrastive loss (vs. uni-directional contrastive loss in the paper).
* *Pose editing:* the provided model uses a transformer-based text encoder (frozen DistilBert + learned transformer), for consistency with the other provided models (vs. GloVe+biGRU configuration used to report results in the paper). Note: this model was finetuned using the best setting as per Table 4: with L/R flip and paraphrases. The FID value may also change as evaluation is carried out with an improved version of the text-to-pose retrieval model.
* *Text generation models*: evaluated with improved retrieval models; also note that, despite an average over 10 repetitions, R-precision metrics come with a great variability due to the randomized selection of the pool of samples to compare against.
</details>

## References in `shortname_2_model_path.txt`

References should be given using the following format:

```
<model shortname><4 spaces><path to the model>
```

Thus, for the above-mentioned models (simply replace `<GENERAL_EXP_OUTPUT_DIR>` by its proper value):
```text
ret_distilbert_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/ret_distilbert_dataPSA2ftPSH2/seed1/checkpoint_best.pth
modret_distilbert_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/modret_distilbert_dataPFAftPFH/seed1/checkpoint_best.pth
gen_distilbert_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/gen_distilbert_dataPSA2ftPSH2/seed1/checkpoint_best.pth
b_gen_distilbert_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/b_gen_distilbert_dataPFAftPFH/seed1/checkpoint_best.pth
capgen_CAtransfPSA2H2_dataPSA2ftPSH2    <GENERAL_EXP_OUTPUT_DIR>/capgen_CAtransfPSA2H2_dataPSA2ftPSH2/seed1/checkpoint_best.pth
modgen_CAtransfPFAHPP_dataPFAftPFH    <GENERAL_EXP_OUTPUT_DIR>/modgen_CAtransfPFAHPP_dataPFAftPFH/seed1/checkpoint_best.pth
```
