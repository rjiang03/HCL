# Hard-Negative Sampling for Contrastive Learning: Optimal Representation Geometry and Neural- vs Dimensional-Collapse

Welcome to the official repository for the paper *"Hard-Negative Sampling for Contrastive Learning: Optimal Representation Geometry and Neural- vs Dimensional-Collapse."* This repository includes implementations of Supervised Contrastive Learning (SCL), Hard-Supervised Contrastive Learning (HSCL), Unsupervised Contrastive Learning (UCL), and Hard-Unsupervised Contrastive Learning (HUCL), along with all the experiments described in the paper.

## Implementations on Image Datasets

- The code for SCL and HSCL can be found in the `SCL` folder.
- The code for UCL and HUCL is located in the `UCL` folder.

### Configurations:
- Set the `estimator` parameter to `"hard"` for the hard version, or `"easy"` to disable the hard negative sampling.
- The `beta` parameter, as defined in the paper, controls the level of hardness of the negative samples.

### Example Usage:

To run HSCL with a beta value of 5, execute the following command inside the `SCL` folder:
```
python main.py --estimator "hard" --beta 5
```

The learned representations and other output data will be saved as `.npy` files in the same folder.

### Reproducing Figures from the Paper:

Once the data is generated, you can reproduce the figures from the paper using the provided Jupyter notebook. This notebook implements the three metrics discussed in the paper.

## Citation

If you find this repository helpful in your research, please consider citing our paper:

```
Citation is coming soon.
```



## Acknowledgements

This code is based on the HCL implementation by [Josh/HCL](https://github.com/joshr17/HCL) and incorporates elements from [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR). Special thanks to both contributors for their foundational work.
