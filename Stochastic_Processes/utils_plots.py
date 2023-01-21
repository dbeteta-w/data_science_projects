import numpy as np
import stochastic_plots as stoch
import matplotlib.pyplot as plt

from scipy.stats import norm


def plot_pmf(X, pmf):
    """ Plots empirical distribution (histogram) using X.
        Plots theoretical distribution using pmf (discrete).

    Parameters
    ----------
    X: ndarray
    Results of M simulations

    pmf: float
    Lambda function of the theoretical distribution


    """

    # Plot histogram of empirical distribution
    fig = plt.figure(1)
    fig.clf()
    n_bins = X.max() - X.min()

    plt.hist(X, bins=n_bins, density=True, edgecolor='black', label='Empirical')
    plt.title('Distribution of Counts of Poisson Process in [0,2] for lambda = 10')
    plt.xlabel('n', fontsize=10)
    plt.ylabel('pmf(n)', fontsize=10)

    # Compare with exact distribution
    x_plot = np.arange(X.min(), X.max(), 1)
    y_plot = [pmf(x) for x in x_plot]
    plt.plot(x_plot, y_plot, linewidth=2, color='r', label='Theoretical')
    plt.legend()


def plot_erlang(times, pdf, n):
    """ Plots the pdf of the time of the n-th event in a sequence of M simulations.

    Parameters
    ----------
    times: list of M lists
    Simulation consisting of M sequences (lists) of arrival times.

    pdf: function
    lambda function of the theoretical distribution

    n: int
    Index of event to plot its time of occurrence distribution
    """

    # n-1 accounts for python indexing (n=1 event is at index 0)
    time_values = [sim[n - 1] for sim in times]

    stoch.plot_pdf(time_values, pdf)
    plt.title('Time occurrence distribution of event n = {}'.format(n))
    plt.show()


def plot_wiener_empiric_and_theoretical_values(empirical_values,
                                               theoretical_values,
                                               integration_grid,
                                               threshold_time):
    """Plot the empirical and theoretical values

    Parameters
    ----------
    empirical_values: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Array of the empirical values simulated

    theoretical_values: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Array of the theoretical values => min(t, threshold_time)

    integration_grid: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Regular grid of discretization times in [initial_time, initial_time+simulation_length]

    threshold_time: float
    Parameter of the gamma function
    """

    plt.plot(
        integration_grid, theoretical_values,
        label="Theoretical value", color='red'
    )
    plt.plot(
        integration_grid, empirical_values,
        label="Empirical value", color='blue'
    )
    plt.axvline(x=threshold_time, color='black', ls="--")
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('$\gamma(W(t)W({}))$'.format(threshold_time))
    plt.title('Empirical vs Theoretical Autocovariance')


def plot_trajectories_and_histogram(process_values,
                                    integration_grid,
                                    process_number,
                                    initial_level,
                                    simulation_length,
                                    max_trajectories,
                                    max_bins):
    """Plot the trajectories of the process values and compare the histogram
    of the final values of the simulated trajectories with the theoretical pdf

    Parameters
    ----------
    process_values: np.ndarray [float]
    Array of the process values obtained

    integration_grid: np.ndarray [float]
    Regular grid of discretization times

    process_number: int
    Number identifier of the process

    initial_level: float
    Starting level of the process

    simulation_length: float
    Length of the simulation

    max_trajectories: int
    Maximum amount of trajectories wanted to be represented

    max_bins: int
    Maximum amount of bins wanted to be represented
    """

    # Trajectories
    limited_trajectories_process_values = process_values[:max_trajectories, :].T
    plt.plot(integration_grid, limited_trajectories_process_values, lw=1)
    plt.title('Simulation')
    plt.xlabel('t')
    plt.ylabel("$V_{}(t)$".format(process_number))
    plt.show()

    # Histogram
    final_process_values = process_values[:, -1]
    plt.hist(final_process_values, bins=max_bins, density=True)
    plt.title('Histogram of the final values vs theoretical pdf')
    plt.xlabel('x')
    plt.ylabel("$V_{}(t)$".format(process_number))

    # Theoretical pdf
    mu, sigma, amount_of_time_steps = 0, 1, 1000
    x_axis_values = np.linspace(
        np.min(final_process_values),
        np.max(final_process_values),
        amount_of_time_steps + 1
    )
    theoretical_pdf = norm.pdf(
        x_axis_values,
        initial_level + mu*simulation_length,
        sigma*np.sqrt(simulation_length)
    )
    plt.plot(x_axis_values, theoretical_pdf, lw=2, color='red')
    plt.show()
