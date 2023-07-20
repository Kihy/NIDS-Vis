# NIDS-Vis
This repository contains code for NIDS-Vis, a black-box traversal algorithm that traverses the decision boundary of black-box Network Intrusion Detection Systems (NIDS)

# Installation
- Download and install docker
- Pull docker image by running: ``docker pull kihy/nids-vis``
- start the docker with: ``docker run --gpus all -it --rm -v "{path to this repo}":/NIDS-Vis kihy/nids-vis bash``

# Running the Code
The code section contains several files and folders.
- KitNET: contains source code Kitsune NIDS, from https://github.com/ymirsky/Kitsune-py
- after_image: contains code for network feature extract of Kitsune
- adv_based_vis.py: contains code for running NIDS-Vis. 
- adv_detect.py: contains code for running robustness experiments
- autoencoder.py: contains code for creating and training autoencoders
- db_characteristics.py: contains code for visualising result of NIDS-Vis and draw decision boundaries
- experiment.py: contains code for training the autoencoders
- feature_partition.py: contains code for feature partitioning
- helper.py: contains helper functions 

Once the dataset is preprocessed and NIDS is trained, edit the files.json and nids_models.json to set up the configs. 
To evaluate the robustness of the NIDSes, edit the configurations in adv_detect.py and run ``python adv_detect.py --command adv_exp``
To evaluate the performance and threshold of the NIDSes, edit the configurations in adv_detect.py and run ``python adv_detect.py --command find_threshold``
To plot the robustness and performance of the NIDSes, edit the configurations in adv_detect.py and run ``python adv_detect.py --command plot_res``
To run NIDS-Vis, edit the configurations in adv_based_vis.py and run ``python adv_based_vis.py``
To plot the results of NIDS-Vis, edit the configurations in db_characteristics.py and run ``python db_characteristics.py``
To apply feature partition to dataset, edit the configurations in feature_partition.py and run ``python feature_partition.py``
