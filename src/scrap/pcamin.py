import numpy as np
from sklearn.decomposition import PCA, KernelPCA, NMF, MiniBatchNMF
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Function to find minimum PCA components for perfect reconstruction
def find_min_pca_components(X, n_splits=5, tolerance=0.999999999):
    """
    Find the minimum number of PCA components needed for (near) perfect reconstruction.

    Parameters:
    X (numpy array): The dataset for which we want to find PCA components.
    n_splits (int): Number of splits for KFold cross-validation.
    tolerance (float): The reconstruction error threshold to consider "perfect" reconstruction.

    Returns:
    min_components (int): The minimum number of PCA components needed for perfect reconstruction.
    """

    # KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterate over each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]

        # Loop over possible PCA components, starting from 1 to the number of features
        for n_components in range(200, X.shape[1] + 1):
            # Apply PCA with 'n_components'
            #pca = PCA(n_components=n_components)
            pca = MiniBatchNMF(n_components=n_components, verbose=True,max_iter=600)
            #pca = KernelPCA(n_components=n_components, kernel="poly", fit_inverse_transform=True)
            X_train_pca = pca.fit_transform(X_train)

            # Reconstruct the data
            X_train_reconstructed = pca.inverse_transform(X_train_pca)

            X_train_reconstructed = np.where(X_train_reconstructed < 0.5, 0, 1)

            # Calculate reconstruction error using Mean Squared Error (MSE)
            mse = accuracy_score(X_train, X_train_reconstructed)
            print(mse, n_components)
            # If MSE is below the tolerance, we assume it's a perfect reconstruction
            if mse > tolerance:
                print(f"Perfect reconstruction with {n_components} components (MSE: {mse})")
                return n_components

    # If no perfect reconstruction found, return the full number of components
    return X.shape[1]

# Example usage with dummy data
if __name__ == "__main__":
    # Generate a random dataset (replace with actual data)
    X = np.random.rand(100, 10)

    # Find the minimum number of PCA components
    min_components = find_min_pca_components(X)
    print(f"Minimum PCA components for perfect reconstruction: {min_components}")
