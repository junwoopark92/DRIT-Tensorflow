# update celebA pretrained model usage
![directory](./assets/directory_shape.PNG)

## pretrained model download (128 x 128 size)
[black2blond](https://drive.google.com/open?id=15rxB2M6bGeVA7JO3DPTwaYqUt5_uGEUm)
[black2brown](https://drive.google.com/open?id=1L8wBMcJECSb4Kcn1UnkC2CuBELLfGgXU)


# DRIT-Tensorflow
Simple Tensorflow implementation of [Diverse Image-to-Image Translation via Disentangled Representations](https://arxiv.org/abs/1808.00948) (ECCV 2018 Oral)

<img src='./assets/final.gif' width="800px">

## Pytorch version
* [Author_pytorch_code](https://github.com/HsinYingLee/DRIT)

## Requirements
* Tensorflow 1.8
* python 3.6

## Usage
### Download Dataset
* [cat2dog](http://vllab.ucmerced.edu/hylee/DRIT/datasets/cat2dog)
* [portrait](http://vllab.ucmerced.edu/hylee/DRIT/datasets/portrait)
* [CycleGAN](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
           
├── guide.jpg (example for guided image translation task)
```
  
### Train
```
python main.py --phase train --dataset summer2winter --concat True
```

### Test
```
python main.py --phase test --dataset summer2winter --concat True --num_attribute 3
```

### Guide
```
python main.py --phase guide --dataset summer2winter --concat True --direction a2b --guide_img ./guide.jpg
```

### Tips
* --concat
  * `True` : for the **shape preserving translation** (summer <-> winter) **(default)**
  * `False` : for the **shape variation translation** (cat <-> dog)
  
* --n_scale
  * Recommend `n_scale = 3` **(default)**
  * Using the `n_scale > 1`, a.k.a. `multiscale discriminator` often gets better results
  
* --n_dis
  * If you use the multi-discriminator, then recommend `n_dis = 4` **(default)**
  * If you don't the use multi-discriminator, then recommend `n_dis = 6`
  
* --n_d_con
  * Author use `n_d_con = 3` **(default)**
  * Model can still generate diverse results with `n_d_con = 1`
  
* --num_attribute **(only for the test phase)**
  * If you use the `num_attribute > 1`, then output images are variously generated

## Summary
### Comparison
![comparison](./assets/comparison.png)

### Architecture
![true](./assets/true.png)
![false](./assets/false.png)

### Train phase
![train_1](./assets/train_1.png)
![train_2](./assets/train_2.png)

### Test & Guide phase
![test](./assets/test.png)

## Results
![result_1](./assets/result1.png)
![result_2](./assets/result2.png)

## Related works
* [UNIT-Tensorflow](https://github.com/taki0112/UNIT-Tensorflow)
* [MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow)

## Author
Junho Kim
