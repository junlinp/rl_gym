import gymnasium as gym
import torch    
from critic_actor import Actor, Critic
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load the trained agent
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='human')

    actor_path = f"trained_agents/{env_name}_actor.pth"
    critic_path = f"trained_agents/{env_name}_critic.pth"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))

    # draw the value function
    state, info = env.reset()
    x_list = []
    y_list = []
    z_list = []
    action_list = []
    for i in range(1):
        while True:
            env.render()
            action_probs = actor(torch.tensor(state))
            action = torch.multinomial(action_probs, 1)

            critic_value = critic(torch.tensor(state))
            next_state, reward, terminated, truncated, info = env.step(action.item())

            x = state[0]
            y = state[2]
            z = critic_value.item()
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            action_list.append(action.item())
            state = next_state
            if terminated or truncated:
                state, info=env.reset()
                break

    # print the min, max of z_list
    print(f"min z: {min(z_list)}, max z: {max(z_list)}")

    # print the action_list
    for i in range(len(action_list)):
        print(f"x {i} : {x_list[i]}, y {i} : {y_list[i]}, z {i} : {z_list[i]}, action {i} : {action_list[i]}")
    # draw the value function, label the x, y, z axis   
    # draw different colors for different z values  to let me know the index of the z values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_list, y_list, z_list, c=z_list, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()  