# About the PoseScript dataset

## :inbox_tray: Download

**License.**
*The PoseScript dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.
A summary of the CC BY-NC-SA 4.0 license is located [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The CC BY-NC-SA 4.0 license is located [here](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).*

| Version | Link | What changed ? |
|---|---|---|
| **V2** | [download](https://download.europe.naverlabs.com/ComputerVision/PoseScript/posescript_release_v2.zip) | pose set is 100k instead of 20k ; file format updated according to the 2023 code release |
| **V1** | [download](https://download.europe.naverlabs.com/ComputerVision/PoseScript/posescript_dataset_release.zip) | 6293 human-written pairs instead of 3893 |
| **V0 (ECCV22)** | (as V1) | |


<details>
<summary>Dataset content.</summary>

* a file linking each pose ID to the reference of its corresponding pose sequence in AMASS, and its frame index;
* a file linking each pose ID with its descriptions (separate files for automatically generated captions and human-written ones);
* files listing pose IDs for each split.

Please refer to the provided README for more details.
</details>


## :crystal_ball: Take a quick look

To take a quick look at the data (ie. look at the pose under different viewpoints, the human-written caption when available, and the automatic captions):

```
streamlit run posescript/explore_posescript.py
```

## :page_with_curl: Generate automatic captions

### Overview of the captioning pipeline

Given a normalized 3D pose, we use posecodes to extract semantic pose information. These posecodes are then selected, merged or combined (when relevant) before being converted into a structural pose description in natural language. Letters ‘L’ and ‘R’ stand for ‘left’ and ‘right’ respectively.

![Captioning pipeline](../../../images/captioning_pipeline.png)

Please refer to the paper and the supplementary material for more extensive explanations.

### Generate captions

To generate automatic captions, please follow these steps:

*Note: the time estimations below are given for 20k poses, but the newest version of PoseScript has 100k poses.*

- **compute joint coordinates for all poses** _(~ 20 min)_
	```
	python posescript/compute_coords.py
	```

- **get and format BABEL labels for poses in PoseScript** _(~ 5 min)_
	```
	python posescript/format_babel_labels.py
	```	

- (optional) **modify captioning data as you see fit**:
	- looking at diagrams on posecode statistics can be helpful to decide on posecode eligibility. To do so, run the following:
		```
		python posescript/captioning.py --action posecode_stats --version_name posecode_stats
		```
	- (re)define posecodes (categories, thresholds, tolerable noise levels, eligibility), super-posecodes, ripple effect rules based on statistics, template sentences and so forth by modifiying *posescript/captioning_data.py*. The data structures are extensively explained in this file, and one can follow some marks (`ADD_VIRTUAL_JOINT`, `ADD_POSECODE_KIND`, `ADD_SUPER_POSECODE`) to add new captioning material. Note that some posecodes were added to be used in the [pipeline that generates automatic modifiers](../posefix/README.md) (search for the `ADDED_FOR_MODIFIERS` marks).


- **generate automatic captions** _(~ 1 min = 20k captions, with 1 cap/pose)_

	*Possible arguments are:*
    - `--saving_dir`: general location for saving generated captions and data related to them (default: *<POSESCRIPT_LOCATION>/generated_captions/*)
    - `--version_name`: name of the caption version. Will be used to create a subdirectory of `--saving_dir` in which to save all files (descriptions & intermediary results). Default is 'tmp'.
	- `--simplified_captions`: produce a simplified version of the captions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word). This configuration is used to generate caption versions E and F from the paper.
    - `--apply_transrel_ripple_effect`: discard some posecodes using ripple effect rules based on transitive relations between body parts.
    - `--apply_stat_ripple_effect`: discard some posecodes using ripple effect rules based on statistically frequent pairs and triplets of posecodes.
    - `--random_skip`: randomly skip some non-essential posecodes (ie. posecodes that were found to be satisfied by more than 6% of the 20k poses considered in PoseScript).
    - `--add_babel_info`: add sentences using information extracted from BABEL.
    - `--add_dancing_info`: add a sentence stating that the pose is a dancing pose if it comes from DanceDB (provided that `--add_babel_info` is also set to True.).

	<br>
	
	<details>
	  <summary>To generate caption versions similar to the latest version of the dataset (posescript-A2):</summary>

	| Version | Command |
	|---------|---------|
	| N2      | `python posescript/captioning.py --version_name captions_n2 --random_skip --simplified_captions` |
	| N6      | `python posescript/captioning.py --version_name captions_n6 --random_skip --add_babel_info --add_dancing_info` |
	| N7      | `python posescript/captioning.py --version_name captions_n7 --random_skip --apply_transrel_ripple_effect --apply_stat_ripple_effect` |

	*Note that some posecodes were added since, for the release of PoseFix and PoseEmbroider.*
	</details>

	<details>
	  <summary>To generate caption versions similar to the ECCV 2022 paper (posescript-A):</summary>

	| Version | Command |
	|---------|---------|
	| A       | `python posescript/captioning.py --version_name captions_A --apply_transrel_ripple_effect --apply_stat_ripple_effect --random_skip --add_babel_info --add_dancing_info` |
	| B       | `python posescript/captioning.py --version_name captions_B --random_skip --add_babel_info --add_dancing_info` |
	| C       | `python posescript/captioning.py --version_name captions_C --random_skip --add_babel_info` |
	| D       | `python posescript/captioning.py --version_name captions_D --random_skip` |
	| E       | `python posescript/captioning.py --version_name captions_E --random_skip --simplified_captions` |
	| F       | `python posescript/captioning.py --version_name captions_F --simplified_captions` |

	*Note that some posecodes were added since, for the release of PoseFix and PoseEmbroider.*
	</details>



## Citation

If you use this code or the PoseScript dataset, please cite the following paper:

```bibtex
@inproceedings{posescript,
  title={{PoseScript: 3D Human Poses from Natural Language}},
  author={{Delmas, Ginger and Weinzaepfel, Philippe and Lucas, Thomas and Moreno-Noguer, Francesc and Rogez, Gr\'egory}},
  booktitle={{ECCV}},
  year={2022}
}
```

Please also remember to follow AMASS' and BABEL's respective citation guideline if you use the AMASS or BABEL data respectively.