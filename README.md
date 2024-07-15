# FragNet

Old implementation of ["Expressivity and Generalization: Using Substructure Biases for GNNs"](https://arxiv.org/pdf/2406.08210v1) (ICML 2024). 
If you want to work with a much nicer model switch to the branch view/publication.



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
More details can be found in the [seml documentation](https://github.com/TUM-DAML/seml?tab=readme-ov-file#get-started).


## Benchmark results

To recreate the benchmark results, use

```bash
seml experiment_name_of_your_choice add experiment/xyz.yaml
seml experiment_name_of_your_choice start
```
where you replace experiment/xyz.yaml with the configuration file for the corresponding dataset.


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
