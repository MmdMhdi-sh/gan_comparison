import torch

from algorithms.base_gan import BaseGAN
from models.critic import Critic
from models.weights import initialize_weights

class WGANGP(BaseGAN):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.critic = Critic().to(device)

        initialize_weights(self.generator)
        initialize_weights(self.critic)

        self.g_optimizer = self.build_optimizer(
            self.generator.parameters(),
            config
        )

        self.c_optimizer = self.build_optimizer(
            self.critic.parameters(),
            config
        )

        self.lambda_gp = config["lambda_gp"]

        self.critic_iterations = config["critic_iterations"]

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)

        alpha = torch.rand(
            batch_size,
            1,
            1,
            1,
            device=self.device
        )

        interpolated = alpha * real + (1 - alpha) * fake

        interpolated.requires_grad_(True) 

        scores = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.reshape(
            batch_size,
            -1
        )

        gradient_norm = gradients.norm(
            2,
            dim=1
        )

        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty
    
    def update_critic(self, real_images):
        self.generator.train()
        self.critic.train()

        self.set_requires_grad(self.critic, True)

        real_images = real_images.to(self.device)

        z = self.sample_noise(real_images.size(0))

        fake_images = self.generator(z).detach()

        real_score = self.critic(real_images)

        fake_score = self.critic(fake_images)

        self.c_optimizer.zero_grad()

        gp = self.gradient_penalty(
            real=real_images, 
            fake=fake_images
        )

        critic_loss = (
            -(real_score.mean())
            + fake_score.mean()
            + self.lambda_gp * gp
        )

        critic_loss.backward()

        self.c_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "gradient_penalty": gp.item()
        }
    
    def update_generator(self, batch_size):
        self.generator.train()
        self.critic.train()

        self.set_requires_grad(self.critic, False)
        
        z = self.sample_noise(batch_size)

        fake_images = self.generator(z)

        fake_score = self.critic(fake_images)

        generator_loss = -fake_score.mean()

        self.g_optimizer.zero_grad()

        generator_loss.backward()

        self.g_optimizer.step()

        self.set_requires_grad(self.critic, True)

        return generator_loss.item()
    
    def train_step(self, real_images):
        critic_logs = {
            "critic_loss": 0,
            "gradient_penalty": 0
        }
        for _ in range(self.critic_iterations):
            logs = self.update_critic(real_images)

            for k in critic_logs:
                critic_logs[k] += logs[k]

        generator_loss = self.update_generator(
            real_images.size(0)
        )

        return {
            "critic_loss": critic_logs["critic_loss"] / self.critic_iterations,
            "gradient_penalty": critic_logs["gradient_penalty"] / self.critic_iterations,
            "generator_loss": generator_loss
        }

    @property
    def plot_groups(self):
        return [
            ["critic_loss", "generator_loss"],
            ["gradient_penalty"]
        ]
    
    @property
    def checkpoint(self):
        return {
            "config": self.config,
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "c_optimizer": self.c_optimizer.state_dict(),
        }
    
    def load_checkpoint(self, checkpoint):
        self.generator.load_state_dict(
            checkpoint["generator"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.g_optimizer.load_state_dict(
            checkpoint["g_optimizer"]
        )

        self.c_optimizer.load_state_dict(
            checkpoint["c_optimizer"]
        )