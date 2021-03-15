# Generalized Contrastive Loss
## Paper
If you use our code please cite our [paper](https://arxiv.org/abs/2103.06638)
```
@article{leyvavallina2021gcl,
  title={Generalized Contrastive Optimization of Siamese Networks for Place Recognition}, 
  author={María Leyva-Vallina and Nicola Strisciuglio and Nicolai Petkov},
  journal={arXiv preprint arXiv:2103.06638},
  year={2021}
  url={https://arxiv.org/abs/2103.06638}
}
```
## How to use this library
### Download the data
1. MSLS: The dataset is available on request [here](https://www.mapillary.com/dataset/places "MSLS"). For the new GT annotations, please contact us.
2. Pittsburgh: The whole dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/ "Pittsburgh") and the train val splits for Pitts30k are available [here](https://www.di.ens.fr/willow/research/netvlad/ "Pitts30k"). 
3. TokyoTM: The dataset is available on request [here](https://www.di.ens.fr/willow/research/netvlad/ "TokyoTM"). 
4. Tokyo 24/7: The dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/ "Tokyo 24/7"). 
5. TB-Places: The dataset is available [here](https://github.com/marialeyvallina/TB_Places "TB-Places"). For the new GT annotations, please contact us.
5. 7Scenes: The dataset is available [here](https://github.com/marialeyvallina/TB_Places "TB-Places"). For the new GT annotations, please contact us.
### Download the models
All our models can be downloaded from [here](https://drive.google.com/drive/folders/1RHxrAj062ZxDp5817t1s4OXGLP_i8JFX?usp=sharing).
### Reproduce our results
#### MSLS
Run the extract_predictions.py script to compute the map and query features, and the top-k prediction. For instance:
```shell
python3 extract_predictions.py --dataset MSLS --root_dir /mydir/MSLS/ --subset val --model_file models/MSLS/MSLS_resnet152_avg_480_GCL.pth --backbone resnet152 --pool avg --norm L2 --image_size 480,640 --batch_size 4
```
This will produce the file results/MSLS/val/MSLS_resnet152_avg_480_GCL_predictions.txt that you should use to evaluate the MSLS_resnet152_avg_480_GCL model in the [MSLS repository](https://github.com/mapillary/mapillary_sls/tree/master/mapillary_sls).


#### TB-Places
Run the extract_predictions.py script to compute the map and query features, and the map-query distances. For instance:
```shell
python3 extract_predictions.py --dataset TB_Places --root_dir /mydir/TB_Places/ --subset W18_W17 --model_file models/TB_Places/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/TB_Places/W18/W18.json --map_idx_file /mydir/TB_Places/W17/W17.json --f_length 512
```
```shell
python3 extract_predictions.py --dataset TB_Places --root_dir /mydir/TB_Places/ --subset W18_map_query --model_file models/TB_Places/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/TB_Places/W18/W18_query.json --map_idx_file /mydir/TB_Places/W18/W18_map.json --f_length 512
```
For obtaining the top-k recall, run the script eval_recallatk.py. By default, the K values are 1,2,3,4,5,10,15,20,25.
```shell
python3 eval_recallatk.py --prediction_distance_file results/TB_Places/W18_W17/resnet34_avg_GCL_distances.npy --gt_file /mydir/TB_Places/W18_W17_gt.h5 
```

```shell
python3 eval_recallatk.py --prediction_distance_file results/TB_Places/W18_map_query/resnet34_avg_GCL_distances.npy --gt_file /mydir/TB_Places/W18_map_query_gt.h5 
```
#### 7Scenes
Run the extract_predictions.py script to compute the map and query features, and the map-query distances. For instance:
```shell
python3 extract_predictions.py --dataset 7Scenes --root_dir /mydir/7Scenes/ --subset heads --model_file models/7Scenes/heads/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/7Scenes/heads/test.json --map_idx_file /mydir/7Scenes/heads/train.json --f_length 512
```
This will produce the file results/7Scenes/heads/resnet34_avg_GCL_distances.npy, which we can use to evaluate the performance of the resnet34_avg_GCL model.

For obtaining the top-k recall, run the script eval_recallatk.py. By default, the K values are 1,2,3,4,5,10,15,20,25.
```shell
python3 eval_recallatk.py --prediction_distance_file results/7Scenes/heads/resnet34_avg_GCL_distances.npy --gt_file /mydir/7Scenes/heads_gt.h5 
```
For obtaining the Average Precision, run the script eval_recallatk.py.

```shell
python3 eval_AP.py --prediction_distance_file results/7Scenes/heads/resnet34_avg_GCL_distances.npy --gt_file /mydir/7Scenes/heads_gt.h5 
```
### Train your own models
Coming soon
### Define your own graded GT
Coming soon
