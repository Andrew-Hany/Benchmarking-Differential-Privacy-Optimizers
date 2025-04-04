# import matplotlib as plt
# import os
# import numpy as np
# import torch


# class Reporting:

#     @staticmethod
#     def load_results(file_path):
#         try:
#             with open(file_path, 'r') as f:
#                 results = json.load(f)
#             return results
#         except Exception as e:
#             print(f"Error loading results: {e}")
#             return None
#     @staticmethod
#     def generate_summary(results):
#         # Calculate the average loss and accuracy for the final epoch
#         final_epoch_losses = results['Training_all_losses'][-1] if results['Training_all_losses'] else []
#         final_epoch_accuracies = results['Training_all_accuracies'][-1] if results['Training_all_accuracies'] else []

#         final_epoch_loss = np.mean(final_epoch_losses) if final_epoch_losses else None
#         final_epoch_accuracy = np.mean(final_epoch_accuracies) if final_epoch_accuracies else None

#         summary = {
#             'Model File Path': results['model_file_path'],
#             'Parameters': results['parameters'],
#             'Final Training Loss': final_epoch_loss,
#             'Final Training Accuracy': final_epoch_accuracy,
#             'Testing Average Loss': results['Testing_average_loss'],
#             'Testing Total Accuracy': results['Testing_total_accuracy'],
#             'Elapsed Time': results['elapsed_time']
#         }
#         return summary