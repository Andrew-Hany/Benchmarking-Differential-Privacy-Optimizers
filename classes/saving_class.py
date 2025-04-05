import json
import os
import torch
import csv
import numpy as np

from .problems_class import _problem_registry
class Saving:
    
    def __init__(self):
        # Dynamically generate the problem mapping
        self.problem_mapping = {
            problem_type: problem_def.problem_name
            for problem_type, problem_def in _problem_registry.items()
        }

    def save_model(self,model, file_path='trained_model.pth'):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the model state dictionary
            torch.save(model.state_dict(), file_path)
            print(f"Model saved as '{file_path}'")
        except Exception as e:
            print(f"Error saving model: {e}")

        
    def get_model_file_path(self,results_directory,problem_type, optimizer_type, parameters):
        directory = os.path.join(results_directory, str(problem_type), str(optimizer_type),str('models'))
        model_file_name = ''
        for key, value in parameters.items():
            if key not in ["problem_type", "optimizer_type"]:
                model_file_name += str(key) + '_' + str(value) + '_'
        
        model_file_name += '.pth'
        return os.path.join(directory, model_file_name)


    def get_json_file_path(self,results_directory,problem_type, optimizer_type, parameters):
        directory = os.path.join(results_directory, str(problem_type), str(optimizer_type),'json')
        json_file_name = ''
        for key, value in parameters.items():
            if key not in ["problem_type", "optimizer_type"]:
                json_file_name += str(key) + '_' + str(value) + '_'
        
        json_file_name += '.json'
        return os.path.join(directory, json_file_name)

  
    def save_results(self,results_directory,model, parameters,  all_train_losses, all_train_accuracies,all_test_losses,all_test_accuracies, elapsed_time):
        
        parameters["problem_type"] = self.problem_mapping[parameters["problem_type"]]
        results_file_path = self.get_json_file_path(results_directory,parameters["problem_type"], parameters["optimizer_type"], parameters)
        model_file_path = self.get_model_file_path(results_directory,parameters["problem_type"], parameters["optimizer_type"], parameters)
        try:
            # Save the model
            self.save_model(model, model_file_path)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
            
            # Prepare the results dictionary
            results = {
                'model_file_path':model_file_path,
                'parameters':parameters,
                'Training_all_losses': all_train_losses,
                'Training_all_accuracies': all_train_accuracies,
                'Testing_all_losses': all_test_losses,
                'Testing_all_accuracies': all_test_accuracies,
                'elapsed_time': elapsed_time
            }
            
            # Save the results to a JSON file
            with open(results_file_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved as '{results_file_path}'")
            return results
        except Exception as e:
            print(f"Error saving results: {e}")


    def convert_json_to_csv(self,results_directory):
        if not os.path.exists(results_directory):
            os.makedirs(results_directory, exist_ok=True)
        csv_file_path = os.path.join(results_directory, 'results.csv')
        
        try:

            # Collect all JSON files in the directory tree
            json_files = []
            for root, _, files in os.walk(results_directory):
                json_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])
            # Initialize a list to hold all rows of data
            all_data = []
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    # Calculate the average loss and accuracy for the final epoch
                    final_epoch_losses = data['Training_all_losses'][-1] if data['Training_all_losses'] else []
                    final_epoch_accuracies = data['Training_all_accuracies'][-1] if data['Training_all_accuracies'] else []

                    final_epoch_loss = np.mean(final_epoch_losses) if final_epoch_losses else None
                    final_epoch_accuracy = np.mean(final_epoch_accuracies) if final_epoch_accuracies else None
                    row = {
                        'model_file_path': data['model_file_path'],
                        # 'Training_all_losses': data['Training_all_losses'],
                        # 'Training_all_accuracies': data['Training_all_accuracies'],
                        'Testing_average_loss': data['Testing_average_loss'],
                        'Testing_total_accuracy': data['Testing_total_accuracy'],
                        'final_epoch_loss':final_epoch_loss,
                        'final_epoch_accuracy':final_epoch_accuracy,
                        'elapsed_time': data['elapsed_time']
                    }
                    row.update(data['parameters'])
                    all_data.append(row)
            
            # Get all unique keys for CSV columns
            fieldnames = set()
            for row in all_data:
                fieldnames.update(row.keys())
            
            # Write to CSV file
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)
            
            print(f"CSV file saved as '{csv_file_path}'")
            return csv_file_path
        
        except Exception as e:
            print(f"Error converting JSON to CSV: {e}")
