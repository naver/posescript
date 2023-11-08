# {Text :left_right_arrow: Pose} Retrieval Model

_:warning: In what follows, command lines are assumed to be launched from `./src/text2pose`._

## Model overview

**Possible inputs**: 3D human pose, pose description.

![Retrieval model](../../../images/retrieval_model.png)

## :crystal_ball: Demo

To look at a ranking of poses (resp. descriptions) referenced in PoseScript by relevance to your own input description (resp. chosen pose), using a pretrained model, run the following:

```
bash retrieval/script_retrieval.sh 'demo' </path/to/model.pth>
```

## :bullettrain_front: Train

:memo: Modify the variables at the top of the bash script to specify the desired model & training options.

Then use the following command:
```
bash retrieval/script_retrieval.sh 'train' <training phase: pretrain|finetune> <seed number>
```

## :dart: Evaluate

Use the following command (test on PoseScript-H2):
```
bash retrieval/script_retrieval.sh 'eval' </path/to/model.pth>
```