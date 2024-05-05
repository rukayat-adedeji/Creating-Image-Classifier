import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import json
import argparse
import numpy as np

def load_checkpoint(filepath):
    # Load the model checkpoint
    checkpoint = torch.load(filepath)
    
    # Create a new instance of the model with the architecture used during training
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier based on the checkpoint
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['input_size'], checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_to_name = checkpoint['cat_to_name']
    
    return model

def process_image(image_path):
    # Load image using PIL
    pil_image = Image.open(image_path)
    
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    pil_image = preprocess(pil_image)

    # Convert to NumPy array
    np_image = np.array(pil_image)
    
    # Transpose the color channel
    np_image = np_image.transpose((0, 2, 1))

    return torch.from_numpy(np_image).float()

def predict(image_path, model, topk=5, gpu=True):
    # Use GPU if specified
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Load and process the image to a 4D tensor
    image = process_image(image_path)
    image = image.to(device).float().unsqueeze(0)

    # Load the model
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)

    # Calculate probabilities and top classes
    probabilities = torch.exp(output)
    top_probabilities, top_classes = probabilities.topk(topk)

    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[int(idx)] for idx in top_classes.cpu().numpy().flatten()]

    return top_probabilities.cpu().numpy().squeeze(), top_classes

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')


    args = parser.parse_args()

    # Load the mapping of categories to real names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the model from the checkpoint
    loaded_model = load_checkpoint(args.checkpoint)

    # Perform prediction
    probabilities, classes = predict(args.image_path, loaded_model, topk=args.top_k, gpu=args.gpu)

    # Print the results
    for i in range(len(probabilities)):
        class_name = cat_to_name[classes[i]]
        print(f"{class_name}: {probabilities[i]:.3f}")
