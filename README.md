# Finding pick point in 2D image

## Requirements
- Linux or MacOS
- CUDA >= 10.2
- Python >= 3.8

### Step
1. Create a conda virtual environment and then activate it.
```
conda create -n py38 python=3.8 -y
conda activate py38
```
2. Install requirements
```
pip install -r requirements.txt
```
## How to run
### To prepare dataset
```
python prepare_data.py
```
### To train the segmentation model
```
python train.py
```
### To evaluate the segmentation model
```
python val.py
```
### To predict pick point in 2D image
```
python predict_2d.py -s [path/to/testdata] -m [path/to/segmentation model] -o [path/to/output]
```
See all parameters in predict_2d.py
## Result
[Result on test data](https://github.com/vuongnp/ml_assignment/tree/main/results/section1)
## Documentation
[Document](https://drive.google.com/file/d/1qxFZOnaYuQ2e8v8xACQZnozvrM-EU8mq/view?usp=drive_link)