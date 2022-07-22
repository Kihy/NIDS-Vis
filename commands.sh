docker run --gpus all -it --rm -v "D:\NIDS-Vis":/NIDS-Vis kihy/nids-vis bash

docker run --gpus all -it --rm -v "D:\NIDS-Vis":/NIDS-Vis -v "D:\mtd_defence":/mtd_defence kihy/nids-vis bash

docker run --gpus all -it --rm -v "D:\NIDS-Vis":/NIDS-Vis -v "D:\mtd_defence":/mtd_defence -p 0.0.0.0:6006:6006 kihy/nids-vis bash

tensorboard --logdir logs/ --host 0.0.0.0
