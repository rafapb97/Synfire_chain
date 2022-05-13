Implementation of a stable Synfire chain with 5 layers and 5 neurons per layer.

There are 3 files:
- 1_Synfire_Chain_Lava.ipynb is a jupyter notebook with the implementation in Lava.

- utils_synfire.py has the DenseLayer model and the plotting functions that are called by 1_Synfire_Chain_Lava.ipynb.

- 2_Synfire_Chain_Brian2.ipynb is a jupyter notebook with the implemenation in Brian2.

The Operative System employed was Windows 11.

To run the files:
Install conda: https://www.anaconda.com/products/distribution

conda create --name synfire_env python = 3.8.1

Install lava according to the directives in: https://lava-nc.org/
Install Brian2 according to the directives in: https://brian2.readthedocs.io/en/stable/introduction/install.html

conda install -c conda-forge nb_conda_kernels
conda install ipykernel

Unzip coding_assignment
Change directory to ~/coding_assignment
Lauch jupyter notebooks using the synfire_env kernel.
Run everything and see notes.
