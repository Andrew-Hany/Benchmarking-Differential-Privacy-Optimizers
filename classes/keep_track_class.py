import pandas as pd
import os

class HyperparameterTracker:
    def __init__(self, results_file, hyperparameters, filename):
        self.results_file = results_file
        self.hyperparameters = hyperparameters
        self.filename = filename
        self.problem_mapping = {
            1: "cifar10_3c3d",
            2: "fashion_mnist_2c2d",
            3: "fashion_mnist_vae",
            4: "mnist_vae"
        }

    def extract_hyperparameters(self):
        # Read the CSV file using pandas
        df = pd.read_csv(self.results_file)
        
        # Extract the specified hyperparameters
        extracted_hyperparameters = df[self.hyperparameters].copy()
        
        # Map problem types to numbers
        problem_type_column = 'problem_type'
        if problem_type_column in df.columns:
            reverse_mapping = {v: k for k, v in self.problem_mapping.items()}
            extracted_hyperparameters.loc[:, problem_type_column] = df[problem_type_column].map(reverse_mapping)
        
        # Save the extracted hyperparameters to the specified filename
        extracted_hyperparameters.to_csv(self.filename, index=False)
        
        return self.filename
    
    
    def has_run(self, **kwargs):
        df = pd.read_csv(self.filename)
        
        # Check if the combination already exists and has been run
        condition = True
        for key, value in kwargs.items():
            condition &= (df[key] == value)
        
        return condition.any()

