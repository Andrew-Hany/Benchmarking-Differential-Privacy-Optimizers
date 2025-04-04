from main_ import main_train_wrapper
#--------------------------------------------------------------------------------------------------
# Running the main function


# # Define the types for each hyperparameter
# types = [str, float, int, int, int, int, float, float, int]

# # Read the Hyperparameter combinations from the file
# with open('combinations.txt', 'r') as f:
#     combinations = [tuple(type_(value) for type_, value in zip(types, line.strip().split(','))) for line in f]

# index = int(os.environ['SLURM_ARRAY_TASK_ID'])
# if index >= len(combinations):
#     raise ValueError("SLURM_ARRAY_TASK_ID is out of range")

# optimizer, learning_rate, batch_size, epsilon, seed, num_epoch, delta, clip_bound, problem = combinations[index]


# print(f"Running with hyperparameters: optimizer={optimizer}, learning_rate={learning_rate}, batch_size={batch_size}, epsilon={epsilon}, seed={seed}, num_epochs={num_epoch}, delta={delta}, clip_bound={clip_bound}, problem={problem}")
# main_train_wrapper(
#         results_directory='results',
#         delta=delta,
#         learning_rate=learning_rate,
#         clip_bound=clip_bound,
#         batch_size=batch_size,
#         num_epochs=num_epoch,
#         target_epsilon=epsilon,
#         problem_type=problem,
#         optimizer_type=optimizer,
#         seed=seed,
# )


main_train_wrapper(
    results_directory='results',
    delta=1e-5,
    learning_rate=0.01,
    clip_bound=1,
    batch_size=1024,
    num_epochs=1,
    target_epsilon=5,
    problem_type=1,
    optimizer_type='sgd',
    seed=1,
)


