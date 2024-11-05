import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    best_next_action = np.max(Q[sprime])
    Q[s, a] = Q[s, a] + alpha * (r + gamma * best_next_action - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(Q.shape[1]))
    else:
        return np.argmax(Q[s])

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    n_epochs = 1000
    max_itr_per_epoch = 500
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            Sprime, R, done, _, info = env.step(A)

            r += R
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

            S = Sprime
            if done:
                break

        print("Épisode #", e, " : Récompense totale = ", r)
        rewards.append(r)

    print("Récompense moyenne sur les épisodes = ", np.mean(rewards))

    plt.plot(rewards)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.title("Récompenses en fonction des épisodes")
    plt.show()

    print("Entraînement terminé.\n")

    env.close()
