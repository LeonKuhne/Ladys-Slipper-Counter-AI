Check out https://github.com/thtrieu/darkflow

### First Time Setup
Open up a terminal and paste the following:
cd darkflow && make

#### Create Directories
sample_img/    - validation images for run.py
bin/           - yolo weights
annotations/
images/

#### Download YOLO weights
These can be found (link somewhere) on the Darkflow GitHub
Add the downloaded weights to darkflow/bin and rename to 'yolo.weights'

#### Testing Data
Add the image to test to 'darkflow/sample_img'
Modify 'darkflow/run.py' to use the new images 'image_name'
Run from terminal:
cd darkflow
python3 run.py

### Files
gen_anchors.py  - prints anchors to use in config
train.py        - train the counter
run.py          - test using image
tracker.py      - test using video (WIP)


### Configuration

#### Adding Classes
darkflow/labels.txt		# add class names here, each on a new line
darkflow/gen_anchors.py		# update 'cfg_classes'
run 'python3 gen_anchors.py'	# note down details to use next
darkflow/cfg/yolo_ls.cfg	# under [region], adjust 'num' and 'anchors',
				# adjust 'filters' from convolutional layer above

#### Adjusting YOLO Image Resolution
darkflow/cfg/yolo_ls.cfg	# adjust (at top) 'width' and 'height' of image 
				# to multiples of 32
darkflow/gen_anchors.py		# match 'cfg_width' and 'cfg_height'

