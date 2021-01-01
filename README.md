# Learning the Predictability of the Future

Code from the paper [Learning the Predictability of the Future](https://arxiv.org/abs/XXXX).

Website of the project in [hyperfuture.cs.columbia.edu](https://hyperfuture.cs.columbia.edu).

This code is built on the DPC code in [github.com/TengdaHan/DPC](https://github.com/geoopt/geoopt).
We also used hyperbolic networks from [github.com/geoopt/geoopt](https://github.com/geoopt/geoopt) and hyperbolic
operations from the [geoopt](https://github.com/geoopt/geoopt) library.

Under `scripts` there are example bash files to run the self-supervised training and finetuning, and the 
supervised training and testing of our model.

You will have to modify the paths to the datasets and to the dataset info folder (read more in the 
[datasets section](#datasets)).

Run python `main.py --help` for information on arguments.

Be sure to have the external libraries in `requirements.txt` installed.

If you use this code, please consider citing the paper as:

```
@article{suris2020hyperfuture,
    title={Learning the Predictability of the Future},
    author={Sur\'is, D\'idac and Liu, Ruoshi and Vondrick, Carl},
    journal={arXiv},
    year={2020}
}
```

## Datasets

We train our framework on four different datasets: [Kinetics600](https://deepmind.com/research/open-source/kinetics), 
[FineGym](https://sdolivia.github.io/FineGym/), [MovieNet](http://movienet.site), and 
[Hollywood2](https://www.di.ens.fr/~laptev/actions/hollywood2/). The data can be downloaded from the original sources.

Other dataset information necessary to run our models (like train/test splits and class hierarchies) can be found in 
[this link (dataset_info.tar.gz)](https://hyperfuture.cs.columbia.edu/dataset_info.tar.gz). This information is in
general the same as in the original datasets, but we provide it to avoid any inconsistencies. You will have to set the 
path to that folder in `--path_data_info`.

As a reminder, you can extract the content from a `.tar.gz` file by using `tar -xzvf archive.tar.gz`.

## Pretrained models

The pretrained models reported in our paper can be found in 
[this link (checkpoints.tar.gz)](https://hyperfuture.cs.columbia.edu/checkpoints.tar.gz):

Each folder (one for each model) contains a `.pth` file with the checkpoint.

To resume training or to pretrain from one of these pretrained models, add the path to that checkpoint to the  
`--resume` or `--pretrain` arguments.

In case there is any doubt or problem, feel free to send us an email.

