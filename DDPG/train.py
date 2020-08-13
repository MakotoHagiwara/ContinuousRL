from IPython.display import display

import matplotlib
import numpy as np
import gym
import torch
from DDPG_Agent import DdpgAgent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from util import get_screen
import pyglet

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def run():
    env = gym.make('Pendulum-v0')

    #検証用にシードを固定する
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    env.seed(42)


    num_episode = 250  # 学習エピソード数（学習に時間がかかるので短めにしています）
    memory_size = 50000  # replay bufferの大きさ
    initial_memory_size = 1000  # 最初に貯めるランダムな遷移の数
    # ログ用の設定
    episode_rewards = []
    num_average_epidodes = 10

    writer = SummaryWriter(log_dir='logs/')
    max_steps = env.spec.max_episode_steps  # エピソードの最大ステップ数
    agent = DdpgAgent(env.observation_space,
                      env.action_space,
                      memory_size=memory_size,
                      writer=writer,
                      is_image = True)

    # 最初にreplay bufferにランダムな行動をしたときのデータを入れる
    state = env.reset()
    for step in range(initial_memory_size):
        pixel = env.render(mode='rgb_array')
        action = env.action_space.sample() # ランダムに行動を選択 
        next_state, reward, done, _ = env.step(action)
        state = deque([get_screen(pixel) for _ in range(3)], maxlen=3)
        agent.memory.add(state, action, reward, done)
        state = env.reset() if done else next_state
    print('%d Data collected' % (initial_memory_size))

    for episode in range(num_episode):
        state = env.reset()  # envからは3次元の連続値の観測が返ってくる
        episode_reward = 0
        for t in range(max_steps):
            pixel = env.render(mode='rgb_array')
            state = deque([get_screen(pixel) for _ in range(3)], maxlen=3)
            print(state)
            action = agent.get_action(state).data.numpy()  #  行動を選択
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.memory.add(state, action, reward, done)
            agent.update()  # actorとcriticを更新
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        writer.add_scalar("reward", episode_reward, episode)
        if episode % 20 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('DDPG: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()

if __name__ == '__main__':
    run()