# MDF-MVSNet

Core code will be uploaded later.

## Training

1. Prepare DTU training set(640x512) and BlendedMVS dataset(768x576).
1. Edit config.py and set "DatasetsArgs.root_dir", "LoadDTU.train_root&train_pair", and "LoadBlendedMVS.train_root".
2. Run the script for training.

```
# DTU
python train.py -d dtu 
# BlendedMVS
python train.py -d blendedmvs
```

## Testing

The Pre-training model in "pth". 

1. Prepare DTU test set(1600x1200) and Tanks and Temples dataset.
2. Edit config.py and set "DatasetsArgs.root_dir", "LoadDTU.eval_root&eval_pair", and "LoadTanks.eval_root"
3. Run the script for the test.

```
# DTU
python eval.py -p pth/dtu_29.pth.pth -d dtu
# Tanks and Temples
python eval.py -p pth/blendedmvs_29.pth -d tans

```


## Fusion

There three methods in "tools": "filter", "gipuma", and "pcd".

### DTU dataset

1. Install fusibile tools: tools/fusibile or https://github.com/kysucix/fusibile
2. Edit tools/gipuma/conf.py and set "root_dir", "eval_folder" and "fusibile_exe_path".
3. Run the script.

```
cd tools/gipuma
python fusion.py -cfmgd
```

### Tanks and Temples dataset

1. Run the script.

```
# filter(main method)
cd tools/filter
python dynamic_filter_gpu.py -e EVAL_OUTPUT_LOCATION -r DATASET_PATH -o OUTPUT_PATH 
# pcd
cd tools/pcd
chmod +x ninja_init.sh
source ninja_init.sh
python fusion.py -e EVAL_OUTPUT_LOCATION -r DATASET_PATH -o OUTPUT_PATH 
```


## Acknowledgements

Thanks to Yao Yao for opening source of his excellent work [MVSNet](https://github.com/YoYo000/MVSNet). 
Thanks to Xiaoyang Guo for opening source of his PyTorch implementation of MVSNet [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch).
Thanks to Jianfeng Yan for opening source of his PyTorch implementation of Dynamic Consistency Checking [D2HC-RMVSNet](https://github.com/yhw-yhw/D2HC-RMVSNet).
Thanks to Jingyang Zhang for opening source of his PyTorch implementation of pcd-fusion [pcd-fusion](https://github.com/jzhangbs/pcd-fusion).