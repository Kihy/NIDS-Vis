# NIDS-Vis
This repository contains code for NIDS-Vis, a black-box traversal algorithm that traverses the decision boundary of black-box Network Intrusion Detection Systems (NIDS)

# Installation
- Download and install docker
- Pull docker image by running: ``docker pull kihy/nids-vis``
- start the docker with: ``docker run --gpus all -it --rm -v "{path to this repo}":/NIDS-Vis kihy/nids-vis bash``

# Running the Code
The code section contains several files and folders
- KitNET: contains source code Kitsune NIDS, from https://github.com/ymirsky/Kitsune-py
- after_image: contains code for network feature extract of Kitsune
- adv_based_vis.py: contains code for running NIDS-Vis. Configurations can be found in the ``if __main__`` section
- adv_detect.py: contains code for running robustness experiments
- autoencoder.py: contains code for creating and training autoencoders
- db_characteristics.py: contains code for visualising result of NIDS-Vis and draw decision boundaries
- experiment.py: contains code for training the autoencoders
- feature_partition.py: contains code for feature partitioning
- helper.py: contains helper functions 
