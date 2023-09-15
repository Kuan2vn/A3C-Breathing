from agent import *
from envi import *
from helper import *

if __name__ == '__main__':
    env = Environment()
    N = 8           # move taken before learning
    batch_size = 32
    n_epochs = 16

    output_folder = 'output_figures/figure'
    output_folder = output_folder + str(env.window_size)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    actor_source_file = "agent/ppo/actor/actor.pth"
    critic_source_file = "agent/ppo/actor/critic.pth"
    destination_folder = "best_model"
    destination_folder2 = "saved_model"
   
    agent = Agent(n_actions=2, batch_size=batch_size,
                    n_epochs=n_epochs,
                    input_dims=np.shape(env.state_slide[1])[0], chkpt_dir='agent/ppo')


    # agent.load_models()

    i = 0

    n_train = 300000


    score_history = []
    plot_loss = []

    avg_score = 0

    best_score = -3
    avg_score1 = 0

    new_number = 0
    save_number = 0

    # env.window_size -= 25

    while i < n_train:

          env.reset()
          observation = env.get_state()
          done = False

          print('Train number ',i)
          while not done:
            action, prob, val = agent.choose_action(observation)
            reward, done, loss = env.step_action(action)        
            agent.remember(observation, action, prob, val, reward, done)
          
          loss1 = agent.learn()

          plot_loss.append(loss)

          if i % 50 == 0 and i != 0:
              agent.save_models()
              avg_score1 = np.mean(score_history[-50:])


          score_history.append(reward)

          avg_score = np.mean(score_history[-1500:])

          # plot_as(score_history[-500:], plot_loss[-500:])

          plot_as(score_history[-500:], plot_loss[-500:])

          # CHANGE WINDOW_SIZE

          if i >= 10000 and avg_score <= -18:
            output_folder = output_folder.replace(str(env.window_size), "")
            env.window_size -= 5
            env.step -= 5
            output_folder = output_folder + str(env.window_size)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            i = 0
            agent.fresh_models()

            _ = copy_file_with_number(actor_source_file, destination_folder2, save_number)
            _ = copy_file_with_number(critic_source_file, destination_folder2, save_number)
            save_number += 1
            



          if avg_score1 > best_score:
            best_score = avg_score1
            _ = copy_file_with_number(actor_source_file, destination_folder, new_number)
            _ = copy_file_with_number(critic_source_file, destination_folder, new_number)
            new_number += 1
            filename = os.path.join(output_folder, f'figure_{i}.png')
            plt.savefig(filename)


          print('window_size: ', env.window_size)

          # SAVE FIGURE

          if i % 500 == 0 and i != 0:
            
            filename = os.path.join(output_folder, f'figure_{i}.png')
            plt.savefig(filename)

          i += 1

          # print('average move taken/ game: ', mean_move_taken)
          print("agent's last 1500 games average score: ", avg_score)
          print('--------------------')