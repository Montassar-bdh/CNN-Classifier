import argparse
import json
from collections import OrderedDict

from train import build_model

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def get_args():
    """Parses command line arguments.

    Returns:
        argparse.Namespace: Object containing parsed arguments.
    """
    
    parser = argparse.ArgumentParser()
    
    # Adds positional arguments
    parser.add_argument('image_path', type=str, 
                        default=r'/home/workspace/ImageClassifier/flowers/test/7/image_07211.jpg', 
                        help='Image to classify')
    parser.add_argument('checkpoint', type=str, default=r'/home/workspace/saved_model/DenseNet_ep19_b32_lr0.001_checkpoint.pth', 
                        help='Inference checkpoint')
    
    # Adds optional arguments
    parser.add_argument('--top_k', type=int, default=1,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=r'/home/workspace/ImageClassifier/cat_to_name.json', 
                        help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, 
                        help='Use GPU for inference')
    return parser.parse_args()


def load_checkpoint(device, filepath):
    """
    Load a saved checkpoint.

    Parameters:
        device (torch.device): Device to load the model on (CPU or GPU).
        filepath (str): Filepath of the saved checkpoint.

    Returns:
        model (torch.nn.Module): Loaded model with the saved weights and architecture.
    """
    
    # Load checkpoint on CPU if device is CPU
    if device == torch.device("cpu"):
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        # Load checkpoint normally if device is GPU
        checkpoint = torch.load(filepath)
    
    # Get architecture name and build model with it
    arch_name = checkpoint['arch']
    model = build_model(arch=arch_name, hidden_units=checkpoint['hyperparams']['hidden_units'],
                        num_classes=len(checkpoint['classes']))
    
    # Load saved weights and class to index mapping
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['classes']
    return model.to(device)


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model and returns it as a Pytorch tensor.

    Args:
        image (PIL.Image): The image to be processed.

    Returns:
        torch.Tensor: The processed image as a Pytorch tensor.
    """
    
    # Set the mean and standard deviation values for normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Create a PyTorch transform pipeline for image preprocessing
    transform = T.Compose([T.Resize(256),           # Resize the image to 256 pixels on the shortest side
                           T.CenterCrop(224),       # Crop the center 224x224 pixels of the image
                           T.ToTensor(),            # Convert the image to a PyTorch tensor
                           T.Normalize(mean, std)   # Normalize the tensor values using the mean and std
                          ])
    # Apply the transform pipeline to the input image and return the resulting tensor
    return transform(image)

def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
        image_path (str): Path of the image to classify
        model (torch.nn.Module): Trained deep learning model to use for classification
        device (torch.device): The device to use for inference (CPU or GPU)
        topk (int): The number of top most likely classes to return
    
    Returns:
        tuple: A tuple containing two elements - a list of probabilities and a list of classes
               corresponding to the topk most likely classes predicted by the model.
    """
    
    # Load the image and pre-process it for the model
    image = Image.open(image_path)
    image_tensor = process_image(image)
    
    # Switch the model to evaluation mode
    model.eval()
    
    # Make a prediction and get the topk probabilities and classes
    topk_pred = model.forward(image_tensor.unsqueeze(0).to(device)).topk(topk)
    probs = F.softmax(topk_pred[0], 1).detach()
    if device == torch.device("cuda:0"):
        probs = probs.cpu()
        
    probs = probs.tolist()[0]
    classes = topk_pred[1].tolist()[0]
    
    # Convert the class indices to actual class labels using the class_to_idx mapping
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    topk_classes = [idx_to_class[c] for c in classes]
    
    return probs, classes

def main():
    """
    This function serves as the entry point for the program. It loads a trained model checkpoint, 
    processes a given image, uses the model to predict the class of the image and outputs the 
    top K (default 1) predicted classes along with their probabilities. The function uses the 
    following arguments:
    
    Args:
        image_path (str): Path of the image file to be classified.
        checkpoint (str): Path of the saved model checkpoint file.
        top_k (int, optional): The number of top classes to be returned as prediction. Default is 1.
        category_names (str): Path of the JSON file mapping the class values to class names.
        gpu (bool, optional): A flag to enable GPU. Default is False.
    
    Returns:
        None
        
    Example Usage:
        python predict.py image_path checkpoint --top_k 5 --category_names cat_to_name.json --gpu
    """
    
    # Parse command line arguments
    args = get_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    
    # Load pretrained model
    pretrained_model = load_checkpoint(device, args.checkpoint)

    # Load class names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Predict class for the given image
    probs, classes = predict(args.image_path, pretrained_model, device, args.top_k)
    
    # Get class names for the predicted classes
    labels = []
    for c in classes:
        labels.append(cat_to_name[str(c)])
        
    # Print the top K predicted classes with their probabilities
    print(f"Top {args.top_k} predictions:")
    for i in range(args.top_k):
        print(f"{labels[i]} (Probability: {probs[i]*100:.2f}%)")

if __name__ == '__main__':
    main()