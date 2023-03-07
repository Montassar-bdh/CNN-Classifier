import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_args():
    """Parses command line arguments and returns parsed arguments.
    
    Returns:
        An argparse.Namespace object containing parsed arguments.
    """
    
    # create argument parser
    parser = argparse.ArgumentParser()
    
    # define arguments
    parser.add_argument('data_dir', type=str, default='./ImageClassifier/flowers', 
                        help = 'Text file with dog names')
    parser.add_argument('--save_dir', type=str, default='/home/workspace/saved_models/', 
                        help = 'Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', choices=('densenet121', 'vgg16', 'vgg13'), 
                        help = 'Choose CNN model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help = 'Model learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, 
                        help = 'Number of neurons in the hidden dense layer')
    parser.add_argument('--epochs', type=int, default=20, 
                        help = 'Number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False, 
                        help = 'Use GPU for training')
    parser.add_argument('--seed', type=int, default=3407, 
                        help = 'Use random seed for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help = 'Set batch size')
    return parser.parse_args()

def prepare_dataset(data_dir, batch_size=32):
    """
    Prepares the image datasets and dataloaders for training, validation, and testing.

    Args:
    	data_dir (str): Directory path where the dataset is stored.
		batch_size (int): Number of images per batch to be loaded into dataloaders.

    Returns:
		image_datasets (dict): A dictionary of ImageFolder datasets for training, validation, and testing.
		dataloaders (dict): A dictionary of DataLoader objects for training, validation, and testing.
    """
    
    # Set the directories for the train, validation, and test datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the mean and standard deviation for image normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Define the data transformations for each dataset
    data_transforms = {'train':T.Compose([T.RandomResizedCrop(224),
                                          T.RandomHorizontalFlip(),
                                          T.ToTensor(),
                                          T.Normalize(mean, std) # Normalize the image
                                         ]),
                       'valid':T.Compose([T.RandomResizedCrop(224),
                                          T.ToTensor(),
                                          T.Normalize(mean, std) # Normalize the image
                                         ]),
                       'test':T.Compose([T.RandomResizedCrop(224),
                                         T.ToTensor(),
                                         T.Normalize(mean, std) # Normalize the image
                                        ])
                      }
    
    # Load the image datasets using the defined transformations
    image_datasets = {'train': ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': ImageFolder(test_dir, transform=data_transforms['test'])
                     }
    
    # Define the dataloaders using the image datasets
    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
               'valid': DataLoader(image_datasets['valid'], batch_size=batch_size),
               'test': DataLoader(image_datasets['test'], batch_size=batch_size)
               }

    return image_datasets, dataloaders

def save_checkpoint(model, optimizer, epochs, batch_size, hidden_units, lr, path):
    """Saves the model checkpoint to disk.

    Args:
        model (torch.nn.Module): The trained model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epochs (int): The number of epochs trained for.
        batch_size (int): The batch size used during training.
        lr (float): The learning rate used during training.
        path (str): The path to save the checkpoint file to.

    Returns:
        None
    """
    # Define the checkpoint dictionary
    checkpoint = {'model_state': model.state_dict(),
                  'arch': model.name,
                  'optimizer_state': optimizer.state_dict(),
                  'classes': model.class_to_idx,
                  'hyperparams': {'epochs': epochs,
                                  'optimizer': optimizer.__class__.__name__,
                                  'batch_size': batch_size,
                                  'hidden_units': hidden_units,
                                  'lr': lr}
                 }
    # Save the checkpoint to disk
    torch.save(checkpoint, f"{path}/{model.__class__.__name__}_ep{epochs}_b{batch_size}_lr{lr}_checkpoint.pth")

def run_epoch(model, criterion, optimizer, scheduler, phase, dataloaders, dataset_sizes, device):
    """Runs a single epoch of training or validation on the given model.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        phase (str): Either "train" or "val" to indicate whether to train or validate the model.
        dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing the data loaders
            for the training, validation, and test sets.
        dataset_sizes (dict[str, int]): A dictionary containing the sizes of the training, validation,
            and test sets.
        device (torch.device): The device on which to perform the computation.

    Returns:
        Tuple[float, float]: The loss and accuracy for the epoch.
    """
    total_loss = 0.0
    correct_preds = 0
    
    # Set the model to either train or eval mode
    model.train() if phase == "train" else model.eval()

    # Iterate over the data loader for the given phase
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        # Set gradients to zero in training phase only
        with torch.set_grad_enabled(phase == "train"):
            # Forward pass to get model predictions
            outputs = model(inputs)
            # Get the predicted class for each input
            _, preds = torch.max(outputs, 1)
            # Compute the batch loss between the predicted and true labels
            batch_loss = criterion(outputs, labels)
            
            # Backpropagate the loss and update the model parameters in training phase only
            if phase == "train":
                batch_loss.backward()
                optimizer.step()
                
        # Compute the total loss for the epoch
        total_loss += batch_loss.item() * inputs.shape[0]
        
        # Compute the number of correct predictions
        correct_preds += torch.sum(preds == labels.data)
    
    # Update the learning rate if in training phase
    if phase == "train":
        scheduler.step()
        
    # Compute the average loss and accuracy for the epoch
    epoch_loss = total_loss / dataset_sizes[phase]
    epoch_acc = correct_preds.double() / dataset_sizes[phase]
    
    # Print the loss and accuracy for the epoch
    print(f"{phase} loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%")
    return epoch_loss, epoch_acc

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    """
    Train the model and validate it with the given hyper-parameters.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train
        criterion (torch.nn.Module): The loss function to optimize
        optimizer (torch.optim.Optimizer): The optimization algorithm to use
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to adjust the learning rate over epochs
        num_epochs (int): The number of epochs to train the model
        dataloaders (dict): A dictionary containing the training, validation and testing DataLoader objects
        dataset_sizes (dict): A dictionary containing the size of the datasets for training, validation and testing
        device (torch.device): The device (CPU/GPU) to use for training and validation
        
    Returns:
        best_model (torch.nn.Module): The PyTorch model with the highest validation accuracy
        best_epoch (int): The epoch number corresponding to the highest validation accuracy
        
    """
    # initialize the best model variables
    best_acc = 0.0
    best_epoch = 0
    best_model = model
    
    # loop over the number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print("*" * 4)
        
        # loop over the training and validation phases
        for phase in ["train", "valid"]:
            epoch_loss, epoch_acc = run_epoch(model, criterion, optimizer, scheduler, phase, dataloaders, dataset_sizes, device)
            
            # early stopping: check if the current validation accuracy is better than the best accuracy so far
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model
                best_epoch = epoch + 1
                
        print("-" * 10) 
        
    # print the best validation accuracy and the corresponding epoch number
    print(f"Best validation Acc: {best_acc * 100:.2f}% (epoch {best_epoch})")

    return best_model, best_epoch

def build_model(arch='densenet121', hidden_units=512, num_classes=102):
    """
    Build and return a pre-trained model based on the specified architecture.

    Args:
        arch (str): Name of the architecture to use. Defaults to 'densenet121'.
        hidden_units (int): Number of units in the hidden layer. Defaults to 512.
        num_classes (int): Number of classes in the dataset. Defaults to 102.

    Returns:
        model: A pre-trained model with a custom classifier.
    """
    
    # Load pre-trained model
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier with custom classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.2)),
                              ('out', nn.Linear(hidden_units, num_classes))
                              ]))
    return model

def main():
    """
    Main entry point for the program. Calls functions to prepare the dataset, build the model, train it, and save the best checkpoint.
    """
    
    # Parse command line arguments
    args = get_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set the device to CPU or GPU depending on availability and user choice
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    
    if torch.cuda.is_available() and args.gpu:
        torch.cuda.manual_seed(args.seed)
    
    # Prepare the dataset and create dataloaders
    image_datasets, dataloaders = prepare_dataset(args.data_dir, batch_size=args.batch_size)
    
    # Build the model with the specified architecture, hidden units, and number of classes
    model = build_model(arch=args.arch, hidden_units=args.hidden_units, num_classes=len(image_datasets['train'].classes))    
    
    # Calculate the size of the training and validation datasets
    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                     'valid': len(dataloaders['valid'].dataset)}
    
    # Set up the optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Move the model to the selected device (CPU or GPU)
    model = model.to(device)
    
    # Train the model and get the best one according to validation accuracy
    best_model, best_epoch = train_model(model, criterion, optimizer, scheduler, args.epochs, dataloaders, dataset_sizes, device)
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makdirs(args.save_dir)
    
    # Save the best model checkpoint
    best_model.name = args.arch
    best_model.class_to_idx = image_datasets['train'].class_to_idx 
    save_checkpoint(best_model, optimizer, best_epoch, args.batch_size, args.hidden_units, args.learning_rate, args.save_dir)
    
if __name__ == '__main__':
    main()