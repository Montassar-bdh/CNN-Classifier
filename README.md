# CNN-Classifier
Udacity final project for classifying Flower images. It uses transfer learning technique 
to train the model.

The project has 2 main parts:
1. the jupyter notebook
2. python scripts: contain the adapted jupyter notebook functions to run smoothly in the terminal.

**NOTE:** to run the codes seamlessly, the dataset should folder should have the following structure:
  ```
  data_dir/
  ├── train
  ├── valid
  └── test
  ```

# Scripts:
the folder contains 
- `train.py`: contains utility functions for :
	- Creating and preprocessing the dataset 
	- Model building, training and testing 
	- Saving checkpoint
  <br/>
  Usage: `python train.py data_dir [args]` where `data_dir` is the dataset location that must respect the above note.
  
  Optional `args`:
    - `--arch`: Choose CNN model architecture: 'densenet121', 'vgg16' or 'vgg13' (Default value: 'densenet')
    - `--save_dir`: Directory to save checkpoints (Default value: '/home/workspace/saved_models/')
    - `--gpu`: Use GPU for training (Default value: False)
    - `--epoch`: Number of training epochs (Default value: 20)
    - `--learning_rate`: Model learning rate (Default value: 0.001)
    - `--hidden_units`: Number of neurons in the hidden dense layer (Default value: 512)
    - `--seed`: Use random seed for training (to provide reproducible results) (Default value: 3407)
    - `--batch_size`: Set batch size (Default value: 32)

  Example:
  `python train.py ./flowers --arch vgg16 --save_dir ./checkpoints --gpu --epoch 30 --learning_rate 0.002 --hidden_units 256 --seed 320 --batch_size 16`

- `predict.py`: contains utility functions for checkpoint loading, model inference and prediction printing:
  <br/>
  Usage: `python test.py image_path checkpoint [args]` where `image_path` is the image to classify and `checkpoint` is the inference checkpoint path.
  
  Optional `args`:
    - `--top_k`: Return top K most likely classes
    - `--category_names`: Use a json file for mapping the categories to real names
    - `--gpu`: Use GPU for inference
    
    Example: `python predict.py ./flowers/test/50.jpg ./checkpoints/test.pth --top_k 5 --category_names cat_to_name.json --gpu`
