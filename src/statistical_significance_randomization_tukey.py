import pickle
import numpy as np
import sys

measure = 'precision_r'
B = 2100000 # Num trials

folder_measures = 'pickles/movielens/measures/final_measures/'


methods = ['Frequency_Baseline', 'RQ1_1', 'RQ1_2', 'RQ1_3a', 'RQ1_3b', 'RQ1_4', 'RQ1_5', 'RQ2_1a', 'RQ2_1b', 'RQ2_2a', 'RQ2_2b']
values_method = []
num_users = []

for m in methods:
    path = folder_measures + measure + '_' + str(m) + '_Final_Remove.pickle'
    # Load measures
    with open(path, 'rb') as handle:
        measure_values = pickle.load(handle)
    # Transform to numpy array
    if isinstance(measure_values, list):
        measure_values = np.array(measure_values)
    # Assign to dictionary
    values_method.append(measure_values)
    num_users.append(len(measure_values))

num_users_unique = set(num_users)
assert len(num_users_unique) == 1
num_users = next(iter(num_users_unique))
print(num_users)

# Build matrix n x m (n: num users, m: num methods)
X = np.zeros((num_users, len(methods)))
for i in range(len(methods)):
    X[:, i] = values_method[i]
X_means = X.mean(axis=0)
#print(X[:20, :])
print(X_means)

# Iterate for B trials
p = np.zeros((len(methods), len(methods)))
for k in range(B):
    # Init n x m matrix X*
    X_p = np.zeros((num_users, len(methods)))
    # Do the permutation for every row
    for j in range(num_users):
        X_p[j, :] = np.random.permutation(X[j, :])
    # Compute differences between max mean row and min mean row
    X_p_means_rows = np.mean(X_p, axis=0)
    q = max(X_p_means_rows) - min(X_p_means_rows)
    for i in range(len(methods)):
        #print(i)
        for j in range(i+1, len(methods)):
            if q > np.abs(X_means[i] - X_means[j]):
                p[i, j] = p[i, j] + (1 / B)
    if k % 1000 == 0:
        print('Iteration ' + str(k))
        sys.stdout.flush()

print(methods)
print(p)





