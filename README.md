# Text2Pose: 3D Human Poses from Natural Language

This repository is the official PyTorch implementation of the paper ["PoseScript: 3D Human Poses from Natural Language"](https://europe.naverlabs.com/research/computer-vision/posescript/), accepted at [ECCV 2022](https://eccv2022.ecva.net/).

![Text2Pose project](./images/main_picture.png)

This code is divided in 3 parts, for the automatic captioning pipeline, the text-to-pose retrieval model and the text-conditioned pose generation model.

The PoseScript dataset, introduced in the paper, contains both human-written pose descriptions collected on Amazon Mechanical Turk and automatic captions; it can be downloaded [here](https://download.europe.naverlabs.com/ComputerVision/PoseScript/posescript_dataset_release.zip).

## Setup


#### :snake: Create python environment

This code was tested in a python 3.7 environment.

From the main code directory:

```
pip install -r requirements.txt
python setup.py develop
```

**Note**: using cuda version 10.2 (please modify *requirements.txt* otherwise).

You may have to run also the following in a python interpreter:
```
import nltk
nltk.download('punkt')
```

#### :inbox_tray: Download data

The PoseScript dataset links human-written descriptions and automatically generated descriptions to poses from the AMASS dataset.

- The PoseScript dataset can be downloaded [here](https://download.europe.naverlabs.com/ComputerVision/PoseScript/posescript_dataset_release.zip).
- The AMASS dataset can be downloaded from [here](https://amass.is.tue.mpg.de/).
- The BABEL dataset can be downloaded from [here](https://babel.is.tue.mpg.de/data.html).
- The SMPL-H body models can be downloaded from [here](https://mano.is.tue.mpg.de/) by clicking on the link _"Extended SMPL+H model"_ on the download page.
- The GloVe pretrained word embeddings can be downloaded [here](https://nlp.stanford.edu/data/glove.840B.300d.zip).


#### :open_file_folder: Define important paths
*:exclamation:Please change paths in ./src/text2pose/config.py following your own preferences.*
- ***GENERAL_EXP_OUTPUT_DIR***: where models will be saved (along with logs, generated poses...)
- ***POSESCRIPT_LOCATION***: where PoseScript is located (vocabulary files will be generated into this directory).
- ***SMPL_BODY_MODEL_PATH***: where SMPL-H body models are located.
- ***AMASS_FILE_LOCATION***: where AMASS is located.
- ***BABEL_LOCATION***: where BABEL is located.
- ***GLOVE_DIR***: where *glove.840B.300d.txt* is located (unzip the downloaded archive).

**Note**: the file *./src/text2pose/shortname_2_model_path.txt* (initially empty) holds correspondences between full model paths and model shortnames, for readable communication between generative and retrieval models. Lines should have the following format:
```
<model_shortname>    <model_full_path>
```


#### :closed_book: Generate the vocabulary
```
cd src/text2pose
python vocab.py --vocab_filename 'vocab3893.pkl' --caption_files 'human3893.json' 'automatic_A.json' 'automatic_B.json' 'automatic_C.json' 'automatic_D.json' 'automatic_E.json' 'automatic_F.json'
```
The vocabury will be saved in ***POSESCRIPT_LOCATION***.

Vocab size is expected to be 1658 and includes 4 special tokens.


## Download and test pretrained models out of the box

To test pretrained retrieval and generative models right out of the box:

- [:inbox_tray: Download pretrained models](https://download.europe.naverlabs.com/ComputerVision/PoseScript/eccv22_posescript_models.zip)
- Unzip the archive and place the content of the resulting directory in ***GENERAL_EXP_OUTPUT_DIR***
- Add the following lines in file *./src/text2pose/shortname_2_model_path.txt* (simply replace `<GENERAL_EXP_OUTPUT_DIR>` by its proper value):
  ```
  ret_glovebigru_vocA1H1_dataA1    <GENERAL_EXP_OUTPUT_DIR>/PoseText_textencoder-glovebigru_vocA1H1_latentD512/train-posescript-A1/BBC/B32_Adam_lr0.0002_stepLR_lrstep20.0_lrgamma0.5/seed0/best_model.pth
  ret_glovebigru_vocA1H1_dataA1ftH1    <GENERAL_EXP_OUTPUT_DIR>/PoseText_textencoder-glovebigru_vocA1H1_latentD512/train-posescript-H1/BBC/B32_Adam_lr0.0002_stepLR_lrstep20.0_lrgamma0.5_pretrained_ret_glovebigru_vocA1H1_dataA1/seed0/best_model.pth
  gen_glovebigru_vocA1H1_dataA1    <GENERAL_EXP_OUTPUT_DIR>/CondTextPoser_textencoder-glovebigru_vocA1H1_latentD32/train-posescript-A1/wloss_kld0.2_v2v4.0_rot2.0_jts2.0_kldnpmul0.02_kldntmul0.0/B32_Adam_lr00001_wd0.0001/seed0/checkpoint_1999.pth
  gen_glovebigru_vocA1H1_dataA1ftH1    <GENERAL_EXP_OUTPUT_DIR>/CondTextPoser_textencoder-glovebigru_vocA1H1_latentD32/train-posescript-H1/wloss_kld0.2_v2v4.0_rot2.0_jts2.0_kldnpmul0.02_kldntmul0.0/B32_Adam_lr1e-05_wd0.0001_pretrained_gen_glovebigru_vocA1H1_dataA1/seed0/checkpoint_1999.pth
  ```
- **Launch a demo**:
  ```
  streamlit run <type>/demo_<type>.py -- --model_path </path/to/model.pth>
  ```
  with:
    - `<type>` being either `retrieval` or `generative`,
    - `</path/to/model.pth>` being any of the model full paths indicated above for *./src/text2pose/shortname_2_model_path.txt*
- **Evaluate the models**:
  ```
  bash <type>/script_<type>.py -a eval -c <model_shortname>
  ```
  with:
    - `<type>` being either `retrieval` or `generative`, 
    - `<shortname>` being any of the model shortnames indicated above for *./src/text2pose/shortname_2_model_path.txt*

  You should obtain something close to:

  | Retrieval model shortname | Data | mRecall | R<sup>P2T</sup>@1 | R<sup>P2T</sup>@5 | R<sup>P2T</sup>@10 | R<sup>T2P</sup>@1 | R<sup>T2P</sup>@5 | R<sup>T2P</sup>@10 |
  |---------------------------|------|---------|-----|-----|------|-----|-----|------|
  | ret_glovebigru_vocA1H1_dataA1 *(seed 0)* | PoseScript-A1 | 74.7 | 47.7 | 78.4 | 87.0 | 58.0 | 85.6 | 91.6 |
  | ret_glovebigru_vocA1H1_dataA1ftH1 *(seed 0)* | PoseScript-H1 | 32.3 | 12.8 | 34.1 | 45.1 | 15.0 | 37.1 | 49.5 |

  | Generative model shortname | Data | FID | ELBO jts| ELBO vert | ELBO rot | mRecall R/G | mRecall G/R |
  |----------------------------|------|-----|---------|-----------|----------|-------------|-------------|
  | gen_glovebigru_vocA1H1_dataA1 *(seed 0)* | PoseScript-A1 | 0.49 | 1.06 | 1.35 | 0.75 | 29.4* | 55.1* |
  | gen_glovebigru_vocA1H1_dataA1ftH1 *(seed 0)* | PoseScript-H1 | 0.48 | 0.52 | 1.11 | 0.48 | 20.0* | 30.4* |

<details>
  <summary>Comments on the results.</summary>

  **Please note that** (*):
  - the computation of the G/R mRecall involves training a new retrieval model,
  - values for G/R and R/G mRecalls can vary, as they involve training or evaluating models on pose samples selected at random, among those generated by the generative model for each caption. The values provided in the table result from a unique random run.

  **The results provided here differ from those in the ECCV version of the paper.** Indeed:
  - Code has changed a bit when we fused the two repositories used initially for pose retrieval and pose generation respectively.
  - We now consider all 52 joints of the SMPL-H model (instead of 24) as input and output; to avoid any problem when comparing the input pose and the generated pose in the loss reconstruction, but also to better model the hands, which are sometimes described by the annotators. Beside, we use the 3D SMPL-H body model from the [human_pose_prior](https://github.com/nghorbani/human_body_prior) library instead of the [smplx](https://github.com/vchoutas/smplx) one, in accordance with the AMASS data.
  - The FID metric is now computed based on the poses generated with samples from the text distribution, instead of samples from the pose distribution.
</details>

## The PoseScript Dataset & the captioning pipeline

To explore the PoseScript dataset and get more details about it or the captioning pipeline, look [here](./src/text2pose/posescript/README.md).

## Text-to-Pose Retrieval models

See :memo: [instructions](./src/text2pose/retrieval/README.md) to train models, then evaluate them quantitatively and qualitatively (demo).

## Text-conditioned Pose Generative models

See :memo: [instructions](./src/text2pose/generative/README.md) to train models, then evaluate them quantitatively and qualitatively (demo).

## Citation

If you use this code or the PoseScript dataset, please cite the following paper:

```
@inproceedings{posescript,
  title={{PoseScript: 3D Human Poses from Natural Language}},
  author={{Delmas, Ginger and Weinzaepfel, Philippe and Lucas, Thomas and Moreno-Noguer, Francesc and Rogez, Gr\'egory}},
  booktitle={{ECCV}},
  year={2022}
}
```

## License

This code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.

Note that some of the softwares to download and install for this project are subject to separate copyright notices and license terms, which use is subject to the terms and conditions under which they are made available; see for instance [VPoser](https://github.com/nghorbani/human_body_prior).