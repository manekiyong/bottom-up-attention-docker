# bottom-up-attention-docker

Adapted from [Vision and Language Group@ MIL](https://github.com/MILVLG/bottom-up-attention.pytorch), with `extract_features.py` along with some other files modified, for compatibility with [Loss_VT_Grounding](https://github.com/drigoni/Loss_VT_Grounding) to substitute fast_rcnn as the visual feature extractor.

Source code of [apex](https://github.com/NVIDIA/apex) and [detectron2](https://github.com/facebookresearch/detectron2/tree/be792b959bca9af0aacfa04799537856c7a92802) has been pre-pulled as well.

## Motivation
The original [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) generates `.npz` files with the following attributes:

`['image_w', 'cls_prob', 'attr_prob', 'bbox', 'num_bbox', 'image_h', 'x']`

However, the original bottom-up-attention is not compatible with newer GPUs, and could only run on CPU. [Vision and Language Group@ MIL](https://github.com/MILVLG/bottom-up-attention.pytorch) reimplemented bottom-up-attention in PyTorch, but the featrue extractor generates `.npz` file with the following attributes: 

`['x', 'bbox', 'num_bbox', 'image_h', 'image_w', 'info']`

This repository dockerizes and modifies the PyTorch reimplementation to align the extracted features to that of the original implementation. Note that the modification is only done for Single-GPU usage.

Files that were modified:
* `src/extract_features.py`
* `src/utils/extract_features_singlegpu.py`
* `src/utils/extract_utils.py`

## File Structure
* Download pretrained model from [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) and place it in `/src` folder. 
* Create `data` folder and copy flickr30k dataset into it.
* Final file structure should be as shown. 

```
.
├── build
│   └── docker-compose.yml
├── data
│   └── flickr30k
│       ├── flickr30k_entities
│       │   ├── Annotations
│       │   ├── Sentences
│       │   ├── test.txt
│       │   ├── train.txt
│       │   └── val.txt
│       ├── flickr30k_feat
│       └── flickr30k_images
├── Dockerfile
├── README.md
└── src
    ├── apex
    ├── bua
    ├── bua-caffe-frcn-r101-k10-100.pth
    ├── configs
    ├── datasets
    ├── detectron2
    ├── evaluation
    ├── extract_features.py
    ├── opts.py
    ├── setup.py
    ├── train_net.py
    └── utils
```

## Usage:
From host:
```
cd build
docker-compose up
```
Once container is up, attach the container's shell and run the following commands:
```
cd /src
python extract_features.py --mode caffe \
         --num-cpus 16 --gpus '0' \
         --extract-mode fast_rcnn \
         --min-max-boxes '10,100' \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --image-dir ../data/flickr30k/flickr30k_images \
         --out-dir ../data/flickr30k/flickr30k_feat
```
Extracted features can be found in the directory indicated in `--out-dir`, and can then be copied to [Loss_VT_Grounding](https://github.com/drigoni/Loss_VT_Grounding) to run `make_dataset_flickr30k.py`