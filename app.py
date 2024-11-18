from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('agg')


app = Flask(__name__)
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Function to generate shifted clusters
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and shift it by `distance`
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 + distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

# Main experiment function
def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    n_samples = 8
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        
        # Calculate slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        
        # Calculate logistic loss
        probs = model.predict_proba(X)[:, 1]
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        loss_list.append(loss)
        
        # Plot the dataset with decision boundary
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
        x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='black', linestyle='--')
        
        plt.title(f"Shift Distance = {distance:.2f}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        
        if i == 1:
            plt.legend(loc='lower right')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices, 
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot parameters vs. shift distance
    plt.figure(figsize=(18, 15))
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list)
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list)
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list)
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list)
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list)
    plt.title("Shift Distance vs Intercept Ratio")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept Ratio")

    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list)
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths)
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    start = float(request.json['start'])
    end = float(request.json['end'])
    step_num = int(request.json['step_num'])

    # Run the experiment with the provided parameters
    do_experiments(start, end, step_num)

    # Check if result images are generated and return their paths
    dataset_img = "results/dataset.png"
    parameters_img = "results/parameters_vs_shift_distance.png"
    
    return jsonify({
        "dataset_img": dataset_img if os.path.exists(dataset_img) else None,
        "parameters_img": parameters_img if os.path.exists(parameters_img) else None
    })

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)
