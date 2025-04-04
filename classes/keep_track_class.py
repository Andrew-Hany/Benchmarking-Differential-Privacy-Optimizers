import pandas as pd
import os
from .problems_class import _problem_registry

class HyperparameterTracker:
    def __init__(self, results_file, hyperparameters, filename):
        self.results_file = results_file
        self.hyperparameters = hyperparameters
        self.filename = filename
        self.problem_mapping = {
            problem_type: problem_definition.__class__.__name__
            for problem_type, problem_definition in _problem_registry.items()
        }


    def extract_hyperparameters(self):
        try:
            df = pd.read_csv(self.results_file)
        except:
            print(f"Error: File not found at {self.results_file}") 
            return None 

        if df.empty:
            print("Warning: The DataFrame is empty. No hyperparameters extracted.")
            empty_df = pd.DataFrame()
            empty_df.to_csv(self.filename, index=False)
            return self.filename

        # Extract the specified hyperparameters
        try:
            extracted_hyperparameters = df[self.hyperparameters].copy()
        except KeyError as e:
            print(f"Error: Hyperparameter(s) not found in DataFrame: {e}")

            empty_df = pd.DataFrame()
            empty_df.to_csv(self.filename, index=False)
            return self.filename
        
        # Map problem types to numbers
        problem_type_column = 'problem_type'
        if problem_type_column in df.columns:
            reverse_mapping = {v: k for k, v in self.problem_mapping.items()}
            extracted_hyperparameters.loc[:, problem_type_column] = df[problem_type_column].map(reverse_mapping)
        
        # Save the extracted hyperparameters to the specified filename
        extracted_hyperparameters.to_csv(self.filename, index=False)
        
        return self.filename
    
    
    def has_run(self, **kwargs):
        try:
            df = pd.read_csv(self.filename)
        except FileNotFoundError:
            return False  # Handle the case where the file doesn't exist

        if df.empty:
            return False  # Return False if the DataFrame is empty
        
        # Check if the combination already exists and has been run
        condition = True
        for key, value in kwargs.items():
            condition &= (df[key] == value)
        
        return condition.any()

