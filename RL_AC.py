# Assume all previous class definitions are correctly imported or defined above
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime

class data_cls:
    def __init__(self, path, test_path=None):
        self.df = pd.read_csv(path, sep=',')
        self.test_df = pd.read_csv(test_path, sep=',') if test_path else None
        self.loaded = True

    def get_data(self, test=False):
        data_df = self.test_df if test else self.df
        shuffled_df = data_df.sample(frac=1).reset_index(drop=True)
        labels = torch.tensor(shuffled_df['labels'].values, dtype=torch.float32)
        features = torch.tensor(shuffled_df.drop('labels', axis=1).values, dtype=torch.float32)
        return features, labels

class RLenv(data_cls):
    def __init__(self, path, test_path=None):
        super().__init__(path, test_path)
        self.state_shape = self.df.shape[1] - 1  # excluding label column

    def reset(self, test=False):
        states, labels = self.get_data(test=test)
        return states, labels

    def step(self, actions, i_iter, states_all, labels_all):
        next_states, labels = states_all[i_iter+1], labels_all[i_iter+1]
        rewards = (actions == labels_all[i_iter]).float() * 1.0
        done = (i_iter+1 >= len(states_all))
        return next_states, rewards, done

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "./dataset/formated_train_simple.data"
    test_path = "./dataset/formated_test_simple.data"
    env = RLenv(train_path, test_path)
    num_actions = 2
    hidden_size = 100
    learning_rate = 0.01
    num_epoch = 1000
    gamma = 0.8

    parent_directory = "result_RLAC"
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    # Create a unique directory for this particular run
    directory_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(parent_directory, directory_name)
    os.makedirs(run_directory)

    actor = Actor(env.state_shape, hidden_size, num_actions).to(device)
    critic = Critic(env.state_shape, hidden_size).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    rewards_history = []
    loss_history = []

    for epoch in range(num_epoch):
        states_all, labels_all = env.reset()
        states_all = states_all.to(device)
        labels_all = labels_all.to(device)
        total_rewards = 0
        total_loss = 0

        for i_iteration in range(len(states_all)//50):
            states = states_all[i_iteration].unsqueeze(0)
            actions_prob = actor(states)
            actions = torch.distributions.Categorical(actions_prob).sample()

            next_states, rewards, done = env.step(actions, i_iteration, states_all, labels_all)
            next_states = next_states.to(device)
            rewards = rewards.to(device)

            state_value = critic(states)
            next_state_value = critic(next_states)

            td_error = rewards + gamma * next_state_value * (1 - done) - state_value
            critic_loss = td_error.pow(2)
            actor_loss = -torch.distributions.Categorical(actions_prob).log_prob(actions) * td_error.detach()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            total_rewards += rewards.item()
            total_loss += (critic_loss.item() + actor_loss.item())


        rewards_history.append(total_rewards/(len(states_all)//50))
        loss_history.append(total_loss / (len(states_all)//50))
        print(f"Epoch {epoch+1}/{num_epoch}, Average Rewards: {total_rewards/(len(states_all)//50)}, Average Loss: {total_loss / (len(states_all)//50):.4f}")

    # Plotting the results
    model_path1 = os.path.join(run_directory, 'modelActor.pth')
    torch.save(actor.state_dict(), model_path1)

    model_path2 = os.path.join(run_directory, 'modelCritic.pth')
    torch.save(critic.state_dict(), model_path2)

    # Plotting and saving results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label='Total Rewards per Epoch')
    plt.title('Rewards History - RL AC')
    plt.xlabel('Epoch')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.savefig(os.path.join(run_directory, 'rewards_history.png'))

    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Average Loss per Epoch')
    plt.title('Loss History - RL AC')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_directory, 'loss_history.png'))
    plt.close()

    # Test phase
    test_states, test_labels = env.reset(test=True)
    test_states = test_states.to(device)
    test_labels = test_labels.to(device)
    predicted_actions = []
    true_labels = []

    with torch.no_grad():
        for i in range(len(test_states)):
            state = test_states[i].unsqueeze(0)
            action_prob = actor(state)
            action = torch.argmax(action_prob, dim=1)
            predicted_actions.append(action.item())
            true_labels.append(test_labels[i].item())

    predicted_actions = np.array(predicted_actions)
    true_labels = np.array(true_labels)

    accuracy = accuracy_score(true_labels, predicted_actions)
    precision = precision_score(true_labels, predicted_actions, average='weighted')
    recall = recall_score(true_labels, predicted_actions, average='weighted')
    f1 = f1_score(true_labels, predicted_actions, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    cm = confusion_matrix(true_labels, predicted_actions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - RL AC')
    plt.savefig(os.path.join(run_directory, 'confusion_matrix.png'))
    plt.close()