import torch
import torch.optim as optim
import torch.nn.functional as F


class Optimizer(object):
    def __init__(self,
                 meta_policy,
                 target_meta_policy,
                 actor,
                 target_actor,
                 critic,
                 target_critic,
                 mini_batch_size,
                 discount,
                 lr,
                 update_epochs):
        self.meta_policy = meta_policy
        self.target_meta_policy = target_meta_policy
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.mini_batch_size = mini_batch_size
        self.discount = discount
        self.update_epochs = update_epochs

        self.epsilon = 1e-8
        self.gamma = 0.9

        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr, eps=self.epsilon)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=self.epsilon)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=self.epsilon)

    def update(self, meta_storage, storage):
        loss_meta = self._update_meta_controller(meta_storage)
        loss = self._update_controller(storge)

        return loss_meta, loss

    def _update_meta_controller(self, meta_storage):

        loss_avg = 0
        n_updates = 0

        if len(meta_storage) < self.mini_batch_size:
            return

        for epoch in range(self.update_epochs):

            data_generator = meta_storage.sample(self.mini_batch_size)

            for sample in data_generator:
                state, goal, reward, next_state, masks = sample
                state = state.squeeze(1)
                goal = goal.long().squeeze(0)
                masks = masks.view(-1)

                current_Q_values = self.meta_policy(state)
                current_Q_values = current_Q_values.gather(1, goal)