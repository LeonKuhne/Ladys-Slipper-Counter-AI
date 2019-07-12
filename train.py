from darkflow.net.build import TFNet

options = {"model": "cfg/yolo_ls.cfg",
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 1000,
           "gpu": 1.0,
           "train": True,
           "annotation": "./annotations/",
           "dataset": "./images/",
           "load": -1
           }

tfnet = TFNet(options)
tfnet.train()
