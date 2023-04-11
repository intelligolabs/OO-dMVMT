# online-gesture-recognition


## Setup paths

Create `data/` if not exists

Create `output/` if not exists

Download dataset and put in `data/shrec22` and `data/shrec19`. Or create symbolic link `ln -s </abs/path/of/data> ./` 

## Train the model
1. In the terminal, run the following command:
```
./train_dMVMT.sh
```
2. In case you want to specify a different configuration, edit the `cfg_path` argument inside `train_dMVMT.sh`
3. The checkpoint will be saved in `output/<exp_name>/best_classifier_model.pth`

## Generate output file for shrec22
1. Open ```eval/eval_shrec22.py```
1. Specify a list of folders containing the model checkpoints you wish to evaluate and the list of corresponding config files (line 178) DEFAULT: `tests=['output/train_shrec22_oo-dmvmt']`, `cfg_paths=['configs/eval_OO-dMVMT.yaml']`
1. From the project root, run the following command in a terminal:
```
PYTHONPATH=. python eval_shrec22.py
```
4. The output file will be saved in `<test_folder>/result.txt`
5. Evaluate the result using the [official script](https://github.com/univr-VIPS/Shrec22) from the hosts of the [Shrec22 challenge](https://univr-vips.github.io/Shrec22/)

## Outline of the configs
All the hyperparameters used for training/testing are contained in `.yaml` files in the `configs` folder
### Train cfg
`train_OO-dMVMT.yaml` contains the configuration for training our OO-dMVMT model.
- `experiment_name` specify the name of the folder the model checkpoint will be saved in e.g. `output/<exp_name>/best_classifier_model.py`
- `save_metrics` specifies whether to save metrics like confusion matrices and classification results (in numpy)
- `do_sdn` enables the SDN head during training (DEFAULT: True)
- `do_gs_ge` enables the GS/GE heads during training (DEFAULT: True)
- `do_gs_ge_OnOff` enables the On-Off strategy for the regression heads (DEFAULT: True)
- `do_OnOff_skip` only train on the windows that contain a gesture start/end. (DEFAULT: False)
- `W` the size of the window (DEFAULT: 16)
- `calc_m` precomputes the JCD, used to speedup training (DEFAULT: True)

### eval cfg
`eval_OO-dMVMT.yaml` contains the configuration for generating the output file for the Shrec22 challenge
- `W` the size of the window. Must be the same used during training (DEFAULT: 16)
- `calc_m` precomputes the JCD, used to speedup training (DEFAULT: False)