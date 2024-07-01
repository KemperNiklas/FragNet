# FragNet

Official implementation of ["Expressivity and Generalization: Using Substructure Biases for GNNs"](https://arxiv.org/pdf/2406.08210v1) (ICML 2024).



## Python environment
First, set up the python environment via

```bash
conda env create -f environment.yml
conda activate fragNet
```
All experiments are managed with [seml](https://github.com/TUM-DAML/seml). So if you have not used seml before, you have to first set up seml via 
```bash
seml configure
```
More details in the [seml documentation](https://github.com/TUM-DAML/seml?tab=readme-ov-file#get-started).


## Benchmark results

To recreate the benchmark results, use

```bash
seml experiment_name_of_your_choice add experiment/xyz.yaml
seml experiment_name_of_your_choice start
```
where you replace experiment/xyz.yaml with the configuration file for the corresponding dataset.

Note that since writing the paper, we slightly updated/simplified the model. Hence, the results might slightly deviate from the published results. These results can be expected:

| Benchmark | Result |
|---- | ---|
|ZINC-10k | 0.0802 ± 0.0037 (MAE) |
|ZINC-full | TODO |
|peptides-struct| 0.2471 ± 0.0005   (MAE)|
|peptides-func| 0.6563 ± 0.0060 (AP)|

To get the exact published results, you can switch to the (more cluttered) branch TODO


## Citation
If you find our work useful, consider citing it.
```
@article{fragment_biases,
  title={Expressivity and Generalization: Fragment-Biases for Molecular GNNs},
  author={Wollschl{\"a}ger, Tom and Kemper, Niklas and Hetzel, Leon and Sommer, Johanna and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2406.08210},
  year={2024}
}
```
