from BM_simulators import *
import math
import arrival_process_simulation as arrival

import numpy as np


def get_counts_per_poisson_process(initial_time,
                                   final_time,
                                   lambda_rate,
                                   amount_of_trajectories):
    """Simulation of homogeneours Poisson process in [initial_time, final_time]
        and return a ndarray of the counts at t1 of each process.
    
    Parameters
    ----------
    initial_time: float
    Initial time for the simulation

    final_time: float
    Final time for the simulation

    lambda_rate: float
    Rate of poisson process

    amount_of_trajectories: int
    Number of times the process is going to be simulated
            
    Returns
    -------
    counts_per_poisson_process: ndarray [float] of shape (amount_of_trajectories,)
    Simulation consisting of M integers of counts N(t=t1).
    """

    times = arrival.simulate_poisson(
        initial_time, final_time, lambda_rate, amount_of_trajectories
    )
    counts_per_poisson_process = np.array([len(process) for process in times])

    return counts_per_poisson_process


def get_pmf_of_num_of_counts(lambda_rate, final_time):
    """Obtain the theoretical probability mass function of the number of counts
    Note that it depends on the amount of arrivals

    Parameters
    ----------
    lambda_rate: float
    Rate of poisson process

    final_time: float
    Final time for the simulation
    """
    pmf_of_num_of_counts = lambda amount_of_arrivals: (1 / math.factorial(amount_of_arrivals)) \
                                    * lambda_rate ** amount_of_arrivals \
                                    * final_time ** amount_of_arrivals \
                                    * np.exp(-lambda_rate * final_time)

    return pmf_of_num_of_counts


def get_wiener_empiric_and_theoretical_values(amount_of_trajectories,
                                              amount_of_time_steps,
                                              initial_time,
                                              threshold_time,
                                              initial_level,
                                              simulation_length):
    """Simulation of the Wiener process in order to
    get the empirical and theoretical values

    Parameters
    ----------
    amount_of_trajectories: int
    Number of times the process is going to be simulated

    amount_of_time_steps: int
    Number of steps of each process

    initial_time: float
    Starting time of the simulation

    threshold_time: float
    Parameter of the gamma function

    initial_level: int
    Starting level of the process

    simulation_length: float
    Length of the simulation

    Returns
    -------
    empirical_values: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Array of the empirical values simulated

    theoretical_values: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Array of the theoretical values => min(t, threshold_time)

    integration_grid: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Regular grid of discretization times in [initial_time, initial_time+simulation_length]
    """

    integration_grid, brownian_simulations = simulate_arithmetic_BM(
        t0=initial_time, B0=initial_level, T=simulation_length,
        mu=0, sigma=1, M=amount_of_trajectories, N=amount_of_time_steps
    )

    empirical_values = np.array(
        [
            _get_autocovariance(
                time_step,
                initial_time,
                threshold_time,
                brownian_simulations,
                simulation_length
            )
            for time_step in integration_grid
        ]
    )

    theoretical_values = np.where(
        integration_grid < threshold_time, integration_grid, threshold_time
    )

    return empirical_values, theoretical_values, integration_grid


def _get_autocovariance(time_step, initial_time, threshold_time,
                        brownian_simulations, simulation_length):
    """Calculate the autocovariance between the current time step
    and the threshold time

    Parameters
    ----------
    time_step: float
    Current value of the iteration of the integration grid

    initial_time: float
    Starting time of the simulation

    threshold_time: float
    Parameter of the gamma function

    brownian_simulations: numpy.ndarray [float] of shape (amount_of_trajectories, amount_of_time_steps+1)
    Simulation consisting of M trajectories where each trajectory is
    a row vector composed of the values of the process at time t

    simulation_length: float
    Length of the simulation

    Returns
    -------
    autocovariance: float
    Value of the autocovariance between the current time step
    and the threshold time
    """

    amount_of_intervals = len(brownian_simulations[0]) - 1

    time_step_index = int(
        (time_step - initial_time) * amount_of_intervals / simulation_length
    )
    final_time_index = int(
        (
                    threshold_time - initial_time) * amount_of_intervals / simulation_length
    )

    autocovariance = np.cov(
        brownian_simulations[:, time_step_index],
        brownian_simulations[:, final_time_index],
    )[0, 1]

    return autocovariance


def get_v1_process_values(amount_of_trajectories,
                          amount_of_time_steps,
                          initial_time,
                          rho_value,
                          initial_level,
                          simulation_length):
    """Simulation of the following process:

    V_1(t) = rho * W(t) + sqrt(1 - rho^2) * W'(t)

    Where t ≥ 0 and W(t) ⊥ W'(t)

    Parameters
    ----------
    amount_of_trajectories: int
    Number of times the process is going to be simulated

    amount_of_time_steps: int
    Number of steps of each process

    initial_time: float
    Starting time of the simulation

    rho_value: float
    Parameter of the process v1

    initial_level: int
    Starting level of the process

    simulation_length: float
    Length of the simulation

    Returns
    -------
    v1_process_values: np.ndarray [float] of shape (amount_of_trajectories, amount_of_time_steps+1)
    Array of the v1 process values obtained

    integration_grid: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Regular grid of discretization times in [initial_time, initial_time+simulation_length]
    """

    integration_grid, first_brownian_simulations = simulate_arithmetic_BM(
        t0=initial_time, B0=initial_level, T=simulation_length,
        mu=0, sigma=1, M=amount_of_trajectories, N=amount_of_time_steps
    )

    _, second_brownian_simulations = simulate_arithmetic_BM(
        t0=initial_time, B0=initial_level, T=simulation_length,
        mu=0, sigma=1, M=amount_of_trajectories, N=amount_of_time_steps
    )

    v1_process_values = rho_value * first_brownian_simulations \
                        + np.sqrt(
        1 - rho_value ** 2) * second_brownian_simulations

    return v1_process_values, integration_grid


def get_v2_process_values(amount_of_trajectories,
                          amount_of_time_steps,
                          initial_time,
                          initial_level,
                          simulation_length):
    """Simulation of the following process:

    V_2(t) = -W(t)

    Where t ≥ 0

    Parameters
    ----------
    amount_of_trajectories: int
    Number of times the process is going to be simulated

    amount_of_time_steps: int
    Number of steps of each process

    initial_time: float
    Starting time of the simulation

    initial_level: int
    Starting level of the process

    simulation_length: float
    Length of the simulation

    Returns
    -------
    v2_process_values: np.ndarray [float] of shape (amount_of_trajectories, amount_of_time_steps+1)
    Array of the v2 process values obtained

    integration_grid: np.ndarray [float] of shape (amount_of_time_steps+1,)
    Regular grid of discretization times in [initial_time, initial_time+simulation_length]
    """

    integration_grid, brownian_simulations = simulate_arithmetic_BM(
        t0=initial_time, B0=initial_level, T=simulation_length,
        mu=0, sigma=1, M=amount_of_trajectories, N=amount_of_time_steps
    )

    v2_process_values = brownian_simulations * -1

    return v2_process_values, integration_grid


def get_v3_process_values(amount_of_trajectories,
                          amount_of_time_steps,
                          initial_time,
                          c_value,
                          initial_level,
                          simulation_length):
    """Simulation of the following process:

    V_3(t) = sqrt(c) * W(t/c)

    Where t ≥ 0 and c > 0

    Parameters
    ----------
    amount_of_trajectories: int
    Number of times the process is going to be simulated

    amount_of_time_steps: int
    Number of steps of each process

    initial_time: float
    Starting time of the simulation

    c_value: float
    Parameter of the process v3

    initial_level: int
    Starting level of the process

    simulation_length: float
    Length of the simulation

    Returns
    -------
    v3_process_values: np.ndarray [float] of shape (amount_of_trajectories, (amount_of_time_steps+1)/c_value)
    Array of the v3 process values obtained

    integration_grid: np.ndarray [float] of shape ((amount_of_time_steps+1)/c_value,)
    Regular grid of discretization times in [initial_time, initial_time+(simulation_length/c_value)]
    """

    integration_grid, brownian_simulations = simulate_arithmetic_BM(
        t0=initial_time, B0=initial_level, T=simulation_length / c_value,
        mu=0, sigma=1, M=amount_of_trajectories, N=amount_of_time_steps
    )

    v3_process_values = np.sqrt(c_value) * brownian_simulations

    return v3_process_values, integration_grid
