import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_adaboost_errors(train_X, train_y, test_X, test_y, n_learners=250, noise=0):
    """
    Train an AdaBoost ensemble and plot training and test errors as a function of the number of fitted learners.

    Parameters
    ----------
    train_X : ndarray of shape (n_samples, n_features)
        Training data features

    train_y : ndarray of shape (n_samples,)
        Training data labels

    test_X : ndarray of shape (n_samples, n_features)
        Test data features

    test_y : ndarray of shape (n_samples,)
        Test data labels

    n_learners : int
        Number of boosting iterations

    noise : float
        Ratio of labels to invert in the dataset
    """
    adaboost = AdaBoost(DecisionStump, n_learners)

    adaboost.fit(train_X, train_y)

    train_errors = []
    test_errors = []

    # Calculate errors for each iteration
    for t in range(1, n_learners + 1):
        train_errors.append(adaboost.partial_loss(train_X, train_y, t))
        test_errors.append(adaboost.partial_loss(test_X, test_y, t))

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, mode='lines', name='Train Error'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_errors, mode='lines', name='Test Error'))

    fig.update_layout(
        title={
            'text': f'Training and Test Errors vs Number of Learners (Noise={noise})',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Number of Learners',
        yaxis_title='Error',
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='LightGray'
        )
    )
    fig.write_html(f"adaboost_errors_noise_{noise}.html")


def plot_decision_boundaries(adaboost, test_X, test_y, T, lims, noise):
    """
    Plot decision boundaries using the ensemble up to specified iterations.

    Parameters
    ----------
    adaboost : AdaBoost
        The fitted AdaBoost classifier

    test_X : ndarray of shape (n_samples, n_features)
        Test data features

    test_y : ndarray of shape (n_samples,)
        Test data labels

    T : list of int
        List of iterations to plot the decision boundary for

    lims : ndarray of shape (2, 2)
        Limits for the plot axes

    noise : float
        Ratio of labels to invert in the dataset
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Iteration {t}" for t in T])

    for i, t in enumerate(T):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(decision_surface(
            lambda X: adaboost.partial_predict(X, t),
            lims[0],
            lims[1],
            density=100,
            dotted=False,
            colorscale=[[0, 'blue'], [1, 'red']],  # Update the colorscale here
            showscale=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                                 marker=dict(color=test_y, colorscale=[[0, 'blue'], [1, 'red']],
                                             # Update the colorscale here
                                             line=dict(color='black', width=1),
                                             colorbar=dict(title="Label")),
                                 showlegend=False),
                      row=row, col=col)

    fig.update_layout(
        title={
            'text': f'Decision Boundaries at Different Iterations (Noise={noise})',
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        paper_bgcolor='lightgray',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    fig.write_html(f"decision_boundaries_noise_{noise}.html")


def plot_best_ensemble(adaboost, test_X, test_y, n_learners=250, lims=None, noise=0):
    """
    Plot decision surface of the best performing ensemble and the test set data points.

    Parameters
    ----------
    adaboost : AdaBoost
        The fitted AdaBoost classifier

    test_X : ndarray of shape (n_samples, n_features)
        Test data features

    test_y : ndarray of shape (n_samples,)
        Test data labels

    n_learners : int
        Number of boosting iterations

    lims : ndarray of shape (2, 2)
        Limits for the plot axes

    noise : float
        Ratio of labels to invert in the dataset
    """
    # Calculate test errors for each iteration
    test_errors = [adaboost.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]

    # Find the iteration with the lowest test error
    best_iter = np.argmin(test_errors) + 1
    best_error = test_errors[best_iter - 1]

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(decision_surface(
        lambda X: adaboost.partial_predict(X, best_iter),
        lims[0],
        lims[1],
        density=100,
        dotted=False,
        colorscale=[[0, 'blue'], [1, 'red']],
        showscale=False
    ))

    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                             marker=dict(color=test_y, colorscale=[[0, 'blue'], [1, 'red']],
                                         line=dict(color='black', width=1),
                                         colorbar=dict(title="Label")),
                             showlegend=False))

    fig.update_layout(
        title={
            'text': f'Best Ensemble Size: {best_iter} with Accuracy: {1 - best_error:.2f} (Noise={noise})',
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        paper_bgcolor='lightgray',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    fig.write_html(f"best_ensemble_decision_boundary_noise_{noise}.html")


def plot_weighted_samples(adaboost, train_X, train_y, lims, noise):
    """
    Plot the training set with point sizes proportional to sample weights and color indicating their labels.

    Parameters
    ----------
    adaboost : AdaBoost
        The fitted AdaBoost classifier

    train_X : ndarray of shape (n_samples, n_features)
        Training data features

    train_y : ndarray of shape (n_samples,)
        Training data labels

    lims : ndarray of shape (2, 2)
        Limits for the plot axes

    noise : float
        Ratio of labels to invert in the dataset
    """
    # Normalize the sample weights for plotting
    D = 5 * (adaboost.D_ / adaboost.D_.max())

    # Create a mesh grid for the decision surface
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], 500),
                         np.linspace(lims[1][0], lims[1][1], 500))
    Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['blue', 'red'])

    scatter = plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, s=D * 30, edgecolor='k', cmap='coolwarm', alpha=0.6)

    # Add color bar
    cbar = plt.colorbar(scatter, ticks=[-1, 1])
    cbar.set_label('Label', rotation=270, labelpad=15)
    cbar.ax.set_yticklabels(['-1', '1'])

    plt.title(f"Final AdaBoost Sample Distribution with Noise Level {noise}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(f"weighted_samples_decision_boundary_noise_{noise}.png", format='png')
    plt.close()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    plot_adaboost_errors(train_X, train_y, test_X, test_y, n_learners, noise)

    # Question 2: Plotting decision surfaces
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    plot_decision_boundaries(adaboost, test_X, test_y, T, lims, noise)

    # Question 3: Decision surface of best performing ensemble
    plot_best_ensemble(adaboost, test_X, test_y, n_learners, lims, noise)

    # Question 4: Decision surface with weighted samples
    plot_weighted_samples(adaboost, train_X, train_y, lims, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
