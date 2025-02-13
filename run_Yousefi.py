import numpy as np
import torch
import torch.nn as nn
import argparse
from Yousefi_Model import load_data_n_model
from widar_model import RecognitionModel, StateMachineModel
from tqdm import tqdm  # Import tqdm for the progress bar

def train(recognition_model, state_machine_model, tensor_loader, num_epochs, learning_rate, criterion, device, sequence_length):
    # Move models to the device (GPU or CPU)
    recognition_model.to(device)
    state_machine_model.to(device)
    
    # Set up the optimizer for both models
    optimizer = torch.optim.Adam(
        list(recognition_model.parameters()) + list(state_machine_model.parameters()), lr=learning_rate
    )
    
    for epoch in range(num_epochs):
        recognition_model.train()
        state_machine_model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        
        # Initialize tqdm progress bar for each epoch
        progress_bar = tqdm(tensor_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()
            
            # Collect features from RecognitionModel across the sequence
            sequence_features = []
            for t in range(sequence_length):
                output = recognition_model(inputs)
                sequence_features.append(output.unsqueeze(1))  # Add time dimension
            
            # Concatenate sequence features along the time dimension
            sequence_features = torch.cat(sequence_features, dim=1)  # Shape: (batch_size, sequence_length, feature_dim)
            
            # Transpose sequence_features to match the expected input shape for Conv1d
            # Shape: (batch_size, feature_dim, sequence_length)
            sequence_features = sequence_features.transpose(1, 2)
            
            # Pass the sequence through StateMachineModel
            outputs = state_machine_model(sequence_features)
            
            # Calculate the loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and accuracy for the epoch
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)
            
            # Update the progress bar with the current loss and accuracy
            progress_bar.set_postfix(loss=loss.item(), accuracy=epoch_accuracy / (len(progress_bar) + 1))
        
        # Calculate average loss and accuracy for the epoch
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.9f}')
    return



def test(recognition_model, state_machine_model, tensor_loader, criterion, device, sequence_length):
    recognition_model.eval()
    state_machine_model.eval()
    test_acc = 0
    test_loss = 0
    
    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            # Collect features from RecognitionModel across the sequence
            sequence_features = []
            for t in range(sequence_length):
                output = recognition_model(inputs)
                sequence_features.append(output.unsqueeze(1))
                
            # Concatenate sequence features along the time dimension
            sequence_features = torch.cat(sequence_features, dim=1)
            
            # Transpose sequence_features to match expected input shape for Conv1d
            sequence_features = sequence_features.transpose(1, 2)  # Shape: (batch_size, feature_dim, sequence_length)
            
            # Pass sequence through StateMachineModel
            outputs = state_machine_model(sequence_features)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            accuracy = (predict_y == labels).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
    
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print(f"Validation accuracy: {test_acc:.4f}, Loss: {test_loss:.5f}")
    return


def main():
    root = './' 
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    
    # Define `dataset` and `model` arguments
    parser.add_argument('--dataset', choices=['Yousefi'], default='Yousefi')
    parser.add_argument('--model', choices=['custom_model'], default='custom_model')
    
    # Define `sequence_length` argument
    parser.add_argument('--sequence_length', type=int, default=10, help="Set sequence length for temporal data")
    
    args = parser.parse_args()
    
    # Load dataset and models using updated load_data_n_model function
    train_loader, test_loader, (recognition_model, state_machine_model), train_epoch = load_data_n_model(args.dataset, args.model, root)
    
    # Define loss criterion and device
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    print(device)
    
    # Train and test
    train(
        recognition_model=recognition_model,
        state_machine_model=state_machine_model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        sequence_length=args.sequence_length  # Use the parsed `sequence_length`
    )
    test(
        recognition_model=recognition_model,
        state_machine_model=state_machine_model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        sequence_length=args.sequence_length  # Use the parsed `sequence_length`
    )

if __name__ == "__main__":
    main()
