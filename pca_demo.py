# ---------------------------------------- DEPENDENCY IMPORTS ---------------------------------------------
import numpy as np  # Utilized to perform operations and store arrays
import copy # Utilized to create complete independent copy of pca_dict
from collections import defaultdict # Utilized to create a dictionary with default values of type 'list'


# ---------------------------------------- CLASS DEFINITION ---------------------------------------------
class NaivePCA_Adjusted:
    def __init__(self, n_components=None, pca_dict=None, explained_variance_=None):
        self.mean_ = None
        # if only n_components are passed -> initialize PCA from scratch
        if n_components:
            self.n_components = n_components
            self.components_ = None
            self.explained_variance_ = None
        # if only pca_dict and explained_variance_ are passed -> initialize per PCA information
        elif pca_dict and explained_variance_:
            self.pca_dict = pca_dict
            self.components_ = pca_dict
            self.explained_variance_ = explained_variance_
        else:
            raise ValueError("Either n_components or both pca_dict and explained_variance_ must be provided")
    
    # Attain corresponding eigenvalues and eigenvectors based on covariance matrix of data (X)
    def fit(self, X):
        # Center the data (so derived relationships are relevant among features)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Store explained variance
        self.explained_variance_ = eigenvalues
        
        # Select top n_components
        if self.n_components is None:
            self.n_components = X.shape[1]
            
        # Set components to the allocated number of eigenvectors to retain
        self.components_ = eigenvectors[:, :self.n_components]
        
        return self
    
    # Apply PCA on data (X) to attain reduced dimensional representation
    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("PCA has not been fitted yet")
            
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
        
    # Run the reduction process
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # Determines the covariance matrix from (modified) components_ and explained_variance_
    def get_covariance_from_components(self):
        # Components are the eigenvectors (PCA components)
        V = self.components_.T  # Transpose to get eigenvectors as columns
        
        # Create diagonal matrix of eigenvalues (PCA explained variance)
        Lambda = np.diag(self.explained_variance_)
    
        # Calculates covariance matrix using: Σ = V Λ V^T
        # Where V is matrix of eigenvectors and Λ is diagonal matrix of eigenvalues
        cov_matrix = V @ Lambda @ V.T
    
        return cov_matrix


# ---------------------------------------- PROCEDURAL FUNCTIONS ---------------------------------------------
# Function to generate PCA objects (designated on 'num_comp' number of components)
def generate_pca(num_comp):
    # Instantiate PCA objects which iteratively reduce the number of principal components by 1
    # Store the generated PCA objects in a dictionary ('pca')
    pca = {}
    for i in range(num_comp-1, 0, -1):      # Generate from num_comp-1 (less than original number of features) to 1
        current_pca = "pca_" + str(i)       # Base the specific PCA object name on the number of principal components
        pca[current_pca] = NaivePCA_Adjusted(n_components=i)  # Generate the PCA object
    return pca

# Function to iteratively reduce the dimensions of original data (X) using the defined PCA objects
def transform_data(pca_dict, X):
    # Iterate over the PCA objects in reverse order (dims reduced in each iteration) to fit 
    # the previous PCA's (higher dimension) output and store the output in 
    # dictionary 'transformed_X'. Effectively, reduces the dimensions of the original data 
    # gradually to allow for fine-grained manipulation of the data on particular features
    transformed_X = {}
    
    # As mentioned, the following is just one of many possible implementations to iteratively reduced
    # the dimensions of the original data
    i = len(list(pca_dict.items()))     # set the starting index to the number of PCA objects
    while i >= 1:
        transformed_X[f"pca_{i}"] = pca_dict[f"pca_{i}"].fit_transform(X)
        X = transformed_X[f"pca_{i}"]
        i -= 1
    return transformed_X

# Function to calculate the variance ratio (percent contribution of each feature for each principal component)
def variance_ratio(pca):
    explained_variance_ratio = {}
    for key, value in pca.items():
        explained_variance_ratio[key] = value.explained_variance_ / np.sum(value.explained_variance_)
    return explained_variance_ratio

# Function to define a threshold of explainability for the components
def threshold_explained_variance(threshold, ratio):
    threshold_metric = {}
    for k, v in ratio.items():  # investigate each explained_variance ratio for each PCA
        elements = 0
        sum = 0.0
        for i in range(len(v)+1):   # consider each ratio
            if sum >= threshold:    # determine if the accumulated variance exceeds the defined threshold
                threshold_metric[k] = sum, elements # if yes, assign the resultant sum along with the number of elements to the specific PCA
            else:
                sum += v[i]         # if no, add the current ratio to the sum
                elements += 1       # increase the elements counter to include current ratio
    return threshold_metric

# Attain the adjusted explained_variance_ and components_ based on the defined threshold
def adjusted_var_pca_dict_explained_variance(pca_dict, thresh):
    num = [num_compo[1] for num_compo in thresh.values()]   # extract the number of components to utilize
    var_pca_explained_variance_ = {}
    var_pca_dict = {}
    for d, n in zip(pca_dict.items(), num):
        # limit per the elements for the specified threshold
        var_pca_explained_variance_.update({d[0] : d[1].explained_variance_[:n]})   
        var_pca_dict.update({d[0] : d[1].components_[:n]})
    return var_pca_explained_variance_, var_pca_dict

# Fit the reduced data from the original PCA on the adjusted PCA components (IMPLEMENTATION DEPENDENT ON sliced_dict)
def data_fit_on_var_pca(fitted_data, sliced_dict):
    fitted_data_on_var = {}
    for i in reversed(range(1,5)):      # apply each adjusted pca to the data reduced on the original PCA constructs
        fitted_data_on_var[f"pca_{i}"] = np.dot(fitted_data[f"pca_{i}"], sliced_dict.components_[f"pca_{i}"].T)
    return fitted_data_on_var

# Function to apply specified weights (0-index reference) to individual PCA components
def adjusted_weighted_components(pca_dict, weights_dict):
    adjusted_dict = copy.deepcopy(pca_dict) # Since its value is itself a dictionary and mutable (use as to not affect original pca_dict)
    try:
        for k, v in pca_dict.items():
            for key, value in weights_dict.items():
                if k == key:
                    # modify the component referenced from the weight_dict key by the scalar as specified by its value
                    adjusted_dict[k].components_[list(value.keys())] = v.components_[list(value.keys())][0] * list(value.values())[0]
    except IndexError as e:
        print(f"***ERROR: {e}. Check that the indices are valid for the specific number of components.***")
    return adjusted_dict

# Function to calculate the covariance matrix based on weighted components and original explained_variance_
def reconstruct_cov_matrix(pca_dict):
    transformed_X = {}

    i = len(list(pca_dict.items()))     
    while i >= 1:
        transformed_X[f"pca_{i}"] = pca_dict[f"pca_{i}"].get_covariance_from_components()
        i -= 1
    return transformed_X

# Fit the reduced data from the original PCA on the weighted PCA components (IMPLEMENTATION DEPENDENT ON adjusted_dict)
def data_fit_on_weights_pca(fitted_data, adjusted_dict):
    fitted_data_on_var = {}
    for i in reversed(range(1,5)):
        fitted_data_on_var[f"pca_{i}"] = np.dot(fitted_data[f"pca_{i}"], adjusted_dict[f"pca_{i}"].components_.T)
    return fitted_data_on_var

# Function to aggregate two PCA objects on their components (IMPLEMENTATION DEPENDENT ON pca_1 and pca_2)
# Only aggregate the minimum number of components for correlation and to avoid errors
def aggregate_pca(pca_1, pca_2):
    agg_pca = defaultdict(list)
    for (k_s, v_s), (k_a, v_a) in zip(pca_1.components_.items(), pca_2.items()):
        if len(k_s) == len(k_a):
            min_compo = min(len(v_s), len(v_a.components_))
            agg_pca[k_s].append((v_s[:min_compo] + v_a.components_[:min_compo]) / 2)
    return agg_pca

# Function to aggregate two PCA objects on their explained variance (IMPLEMENTATION DEPENDENT ON pca_1 and pca_2)
# Only aggregate the minimum number of components for correlation and to avoid errors
def aggregate_explained_variance(pca_1, pca_2):
    agg_explained_variance = defaultdict(list)
    for (k_s, v_s), (k_a, v_a) in zip(pca_1.explained_variance_.items(), pca_2.items()):
        if len(k_s) == len(k_a):
            min_compo = min(len(v_s), len(v_a.explained_variance_))
            agg_explained_variance[k_s].append((v_s[:min_compo] + v_a.explained_variance_[:min_compo]) / 2)
    return agg_explained_variance





# ---------------------------------------- MAIN FUNCTION ---------------------------------------------
if __name__ == "__main__":
    
    # Set seed for reproducible results
    np.random.seed(42)

    # Generate sample data (for demonstration purposes)
    n = 5                       # Number of features (original dimension of data)
    X = np.random.randn(100, n) # (100 samples, n features)
    print("********** ORIGINAL DATA (1st 5 samples) **********")
    print(X[:5])                # View first 5 samples

    # Iterate through the reduction process to attain the relevant explained variance and corresponding components
    pca_dict = generate_pca(n)
    fitted_data = transform_data(pca_dict, X)
    explained_variance_ratio = variance_ratio(pca=pca_dict)

    # View the shape of the data recursively reduced on each PCA construct
    for k, v in fitted_data.items():
        print(f"Shape of data through {k}: {v.shape}")

    # View the components of each pca
    print("********** ORIGINAL PCA COMPONENTS **********")
    for key, value in pca_dict.items():
        print(key)  # prints name relate to number of components (ex:pca_4, pca_3, pca_2, pca_1)
        print(value.components_)    # prints components of each pca

    # View the explained variance ratio of each pca
    print("********** ORIGINAL EXPLAINED VARIANCE RATIO **********")
    for k, v in explained_variance_ratio.items():
        print(k, v)

    print("********** THRESHOLD OF EXPLAINABILITY: 70% **********")
    thresh_70 = threshold_explained_variance(threshold=0.70, ratio=explained_variance_ratio)
    print(thresh_70)

    print("********** ADJUSTED PCA BASED ON THRESHOLD **********")
    var_pca_explained_variance_, var_pca_dict = adjusted_var_pca_dict_explained_variance(pca_dict, thresh_70)
    print(var_pca_explained_variance_)
    print(var_pca_dict)

    print("********** MODIFIED PCA AS NaivePCA_Adjusted OBJECT **********")
    # Initialize the adjusted PCA dict as a NaivePCA_Adjusted Object (with modified constructor)
    sliced_dict = NaivePCA_Adjusted(pca_dict=var_pca_dict, explained_variance_=var_pca_explained_variance_)
    print(f"Adjusted Explained Variance: {sliced_dict.explained_variance_}")
    print(f"Adjusted Components: {sliced_dict.components_}")

    print("********** REDUCED DATA THROUGH THE ADJUSTED PCA (1st 5 through each PCA) **********")
    data_through_var = data_fit_on_var_pca(fitted_data, sliced_dict)
    for k, v in data_through_var.items():
        print(k, v[:5])

    print("********** WEIGHTED PCA COMPONENTS **********")
    # Weight Dictionary defined in format - {key: {component (0-based index): scalar to apply}}
    weights_dict = {'pca_3' : {2: 7}, 'pca_4' : {0: 2}}
    adjusted_weighted_dict = adjusted_weighted_components(pca_dict=pca_dict, weights_dict=weights_dict)
    for k, v in adjusted_weighted_dict.items():
        print(k, v.components_)

    print("********** COVARIANCE MATRIX OF WEIGHTED COMPONENTS **********")
    reconstructed_cov_matrix = reconstruct_cov_matrix(pca_dict=adjusted_weighted_dict)
    print(reconstructed_cov_matrix)

    print("********** REDUCED DATA THROUGH THE WEIGHTED PCA (1st 5 through each PCA) **********")
    data_through_weights = data_fit_on_weights_pca(fitted_data, adjusted_dict=adjusted_weighted_dict)
    for k, v in data_through_weights.items():
        print(k, v[:5])

    print("********** AGGREGATED PCA COMPONENTS **********")
    aggre_pca = aggregate_pca(pca_1=sliced_dict, pca_2=adjusted_weighted_dict)
    for k, v in aggre_pca.items():
        print(k, v[0])

    print("********** AGGREGATED EXPLAINED VARIANCE **********")
    aggre_explained_variance_ = aggregate_explained_variance(pca_1=sliced_dict, pca_2=adjusted_weighted_dict)
    for k, v in aggre_explained_variance_.items():
        print(k, v[0])