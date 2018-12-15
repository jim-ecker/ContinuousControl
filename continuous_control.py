import numpy as np
import torch
import gym
import click
import titlecase
import sys
from collections import deque
import random
from unityagents import UnityEnvironment
from agents.ddpg import Agent
import matplotlib.pyplot as plt

@click.command()
@click.option(
    "--version",
    default=1,
    help   = """
        Select version of environment to run 
        
        1: single agent 
        
        2: multi agent
        
        """
)
@click.option(
    "--env-dir",
    default = "environments",
    help    = """
        Set directory containing environment(s)
    """
)
def run(version, env_dir, n_episodes=2000, max_t=700): # not sure about n_episodes or max_t defaults
    env = load_env(env_dir, version)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    num_agents = len(env_info.agents)
    print('Num Agents: {}'.format(num_agents))

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action: {}'.format(action_size))

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    agent = load_agent(state_size, action_size)
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range (1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
              end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores


def load_agent(state_size, action_size, seed=1337):
    return Agent(state_size=state_size, action_size=action_size, random_seed=seed)


def load_env(env_dir, version):
    filename = None
    if sys.platform == 'linux':
        filename = 'Reacher_Linux_v{}/Reacher.x86_64'.format(version)
    elif sys.platform == 'osx':
        filename = 'Reacher_OSX_v{}.app'.format(version)
    if filename is not None:
        return UnityEnvironment(file_name='{}/{}'.format(env_dir, filename))
    else:
        raise NameError('Couldn\'t parse environment')


if __name__ == '__main__':
    scores = run()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
