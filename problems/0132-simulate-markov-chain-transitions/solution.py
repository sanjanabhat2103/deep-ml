import numpy as np

def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    """
    Simulate a Markov chain.

    Args:
        transition_matrix: Square matrix where transition_matrix[i][j]
                           is the probability of transitioning from
                           state i to state j.
        initial_state: Starting state index.
        num_steps: Number of steps to simulate.

    Returns:
        List of visited states, including the initial state.
    """
    current_state = initial_state
    states = [current_state]

    for _ in range(num_steps):
        current_state = np.random.choice(
            len(transition_matrix),
            p=transition_matrix[current_state]
        )
        states.append(current_state)

    return states