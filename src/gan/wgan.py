"""
Wasserstein GAN (WGAN) Implementation for Crypto Trading Data

Advanced WGAN implementation with improved stability and training dynamics.
Designed for high-quality crypto trading data synthesis with Enterprise patterns.

Features:
- Wasserstein distance for improved stability
- Gradient penalty (WGAN-GP) for better convergence
- Spectral normalization support
- Advanced crypto market modeling
- Production-ready monitoring and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

from loguru import logger
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .vanilla_gan import CryptoDataset, GANConfig

# Configure logging
logger.add(
    "logs/wgan_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


@dataclass
class WGANConfig(GANConfig):
    """Расширенная конфигурация для WGAN"""
    # WGAN специфичные параметры
    critic_iterations: int = 5  # Количество итераций критика на одну итерацию генератора
    gradient_penalty_weight: float = 10.0  # λ для gradient penalty
    use_gradient_penalty: bool = True  # Использовать WGAN-GP
    use_spectral_norm: bool = False  # Спектральная нормализация
    
    # Clipping parameters (для обычного WGAN)
    weight_clip_value: float = 0.01
    
    # Learning rates (часто разные для WGAN)
    critic_lr: float = 0.0001
    generator_lr: float = 0.0001


def spectral_norm_conv(module, use_spectral_norm=True):
    """Применение спектральной нормализации к конволюционному слою"""
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module


def spectral_norm_linear(module, use_spectral_norm=True):
    """Применение спектральной нормализации к линейному слою"""
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module


class WGANGenerator(nn.Module):
    """
    WGAN Generator с поддержкой спектральной нормализации
    и crypto-specific архитектурой
    """
    
    def __init__(self, config: WGANConfig):
        super(WGANGenerator, self).__init__()
        self.config = config
        
        layers = []
        
        # Входной слой
        in_features = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            linear = nn.Linear(in_features, hidden_dim)
            linear = spectral_norm_linear(linear, config.use_spectral_norm)
            
            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Выходной слой (без активации для WGAN)
        output_linear = nn.Linear(in_features, config.sequence_length * config.output_dim)
        output_linear = spectral_norm_linear(output_linear, config.use_spectral_norm)
        layers.append(output_linear)
        # Убираем Tanh для WGAN - пусть генератор производит любые значения
        
        self.main = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"WGAN Generator initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Инициализация весов для WGAN"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, noise):
        output = self.main(noise)
        return output.view(-1, self.config.sequence_length, self.config.output_dim)


class WGANCritic(nn.Module):
    """
    WGAN Critic (Discriminator) с Wasserstein distance
    
    Критик оценивает "реалистичность" данных без sigmoid активации
    """
    
    def __init__(self, config: WGANConfig):
        super(WGANCritic, self).__init__()
        self.config = config
        
        layers = []
        
        # Входной слой
        in_features = config.sequence_length * config.output_dim
        for hidden_dim in reversed(config.hidden_dims):
            linear = nn.Linear(in_features, hidden_dim)
            linear = spectral_norm_linear(linear, config.use_spectral_norm)
            
            layers.extend([
                linear,
                nn.LayerNorm(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Выходной слой (без активации - критик возвращает реальные значения)
        output_linear = nn.Linear(in_features, 1)
        output_linear = spectral_norm_linear(output_linear, config.use_spectral_norm)
        layers.append(output_linear)
        
        self.main = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"WGAN Critic initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Инициализация весов для WGAN Critic"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input):
        input_flat = input.view(-1, self.config.sequence_length * self.config.output_dim)
        return self.main(input_flat)


class WGAN:
    """
    Wasserstein GAN для генерации криптовалютных данных
    
    Implements:
    - Wasserstein distance instead of JS divergence
    - Gradient penalty for improved training stability
    - Critic training multiple times per generator update
    - Enterprise monitoring and logging
    """
    
    def __init__(self, config: WGANConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Инициализация моделей
        self.generator = WGANGenerator(config).to(self.device)
        self.critic = WGANCritic(config).to(self.device)
        
        # Разные learning rates для генератора и критика
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr,
            betas=(config.beta1, config.beta2)
        )
        
        # RMSprop часто работает лучше для WGAN критика
        self.c_optimizer = optim.RMSprop(
            self.critic.parameters(),
            lr=config.critic_lr
        )
        
        # Метрики
        self.training_history = {
            'g_loss': [],
            'c_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': [],
            'epoch': [],
            'timestamp': []
        }
        
        logger.info(f"WGAN initialized on {self.device}")
        logger.info(f"Using gradient penalty: {config.use_gradient_penalty}")
        logger.info(f"Critic iterations: {config.critic_iterations}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Настройка устройства для обучения"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        
        return torch.device(device)
    
    def _gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление gradient penalty для WGAN-GP
        
        Args:
            real_data: Реальные данные
            fake_data: Сгенерированные данные
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_data.size(0)
        
        # Случайные веса для интерполяции
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        # Интерполированные образцы
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Критик для интерполированных данных
        critic_interpolated = self.critic(interpolated)
        
        # Вычисление градиентов
        gradients = grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Градиенты в batch dimension
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty (стремимся к норме градиента = 1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return penalty
    
    def _clip_critic_weights(self):
        """Обрезание весов критика (для обычного WGAN без GP)"""
        if not self.config.use_gradient_penalty:
            for param in self.critic.parameters():
                param.data.clamp_(-self.config.weight_clip_value, self.config.weight_clip_value)
    
    def train(
        self,
        dataloader: DataLoader,
        validation_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Обучение WGAN модели
        
        Args:
            dataloader: DataLoader для обучающих данных
            validation_loader: DataLoader для валидационных данных
            
        Returns:
            История обучения
        """
        logger.info(f"Starting WGAN training for {self.config.num_epochs} epochs")
        
        self.generator.train()
        self.critic.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            epoch_wasserstein_distance = 0.0
            epoch_gradient_penalty = 0.0
            
            for batch_idx, real_data in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Обучение критика несколько раз
                for _ in range(self.config.critic_iterations):
                    c_loss, wasserstein_dist, gp = self._train_critic(real_data, batch_size)
                    epoch_c_loss += c_loss
                    epoch_wasserstein_distance += wasserstein_dist
                    epoch_gradient_penalty += gp
                
                # Обучение генератора один раз
                g_loss = self._train_generator(batch_size)
                epoch_g_loss += g_loss
                
                # Логирование
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"C_loss: {c_loss:.4f} G_loss: {g_loss:.4f} "
                        f"W_dist: {wasserstein_dist:.4f}"
                    )
            
            # Усреднение метрик за эпоху
            num_batches = len(dataloader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_c_loss = epoch_c_loss / (num_batches * self.config.critic_iterations)
            avg_wasserstein_distance = epoch_wasserstein_distance / (num_batches * self.config.critic_iterations)
            avg_gradient_penalty = epoch_gradient_penalty / (num_batches * self.config.critic_iterations)
            
            # Сохранение метрик
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['c_loss'].append(avg_c_loss)
            self.training_history['wasserstein_distance'].append(avg_wasserstein_distance)
            self.training_history['gradient_penalty'].append(avg_gradient_penalty)
            self.training_history['epoch'].append(epoch)
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Валидация
            if validation_loader and epoch % self.config.validate_interval == 0:
                self._validate(validation_loader)
            
            # Сохранение модели
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"wgan_checkpoint_epoch_{epoch}.pth")
            
            logger.info(
                f"Epoch [{epoch}/{self.config.num_epochs}] completed - "
                f"Avg C_loss: {avg_c_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}, "
                f"W_distance: {avg_wasserstein_distance:.4f}"
            )
        
        logger.info("WGAN training completed successfully")
        return self.training_history
    
    def _train_critic(self, real_data: torch.Tensor, batch_size: int) -> Tuple[float, float, float]:
        """Обучение критика"""
        self.c_optimizer.zero_grad()
        
        # Оценка реальных данных
        real_output = self.critic(real_data)
        real_loss = torch.mean(real_output)
        
        # Генерация и оценка фейковых данных
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_data = self.generator(noise).detach()
        fake_output = self.critic(fake_data)
        fake_loss = torch.mean(fake_output)
        
        # Wasserstein distance (приблизительно)
        wasserstein_distance = real_loss - fake_loss
        
        # Gradient penalty (если используется)
        gradient_penalty = 0.0
        if self.config.use_gradient_penalty:
            gp = self._gradient_penalty(real_data, fake_data)
            gradient_penalty = gp.item()
        else:
            gp = 0
        
        # Общий loss для критика (максимизируем Wasserstein distance, минимизируем GP)
        c_loss = fake_loss - real_loss + self.config.gradient_penalty_weight * gp
        c_loss.backward()
        self.c_optimizer.step()
        
        # Обрезание весов (если не используем GP)
        self._clip_critic_weights()
        
        return c_loss.item(), wasserstein_distance.item(), gradient_penalty
    
    def _train_generator(self, batch_size: int) -> float:
        """Обучение генератора"""
        self.g_optimizer.zero_grad()
        
        # Генерация fake data
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_data = self.generator(noise)
        
        # Получение оценки критика
        fake_output = self.critic(fake_data)
        
        # Generator loss (максимизируем оценку критика для fake данных)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def _validate(self, validation_loader: DataLoader) -> Dict[str, float]:
        """Валидация модели"""
        self.generator.eval()
        self.critic.eval()
        
        val_g_loss = 0.0
        val_c_loss = 0.0
        val_wasserstein_distance = 0.0
        
        with torch.no_grad():
            for real_data in validation_loader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Critic loss
                real_output = self.critic(real_data)
                noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.critic(fake_data)
                
                c_loss = torch.mean(fake_output) - torch.mean(real_output)
                wasserstein_dist = torch.mean(real_output) - torch.mean(fake_output)
                val_c_loss += c_loss.item()
                val_wasserstein_distance += wasserstein_dist.item()
                
                # Generator loss
                g_loss = -torch.mean(fake_output)
                val_g_loss += g_loss.item()
        
        val_g_loss /= len(validation_loader)
        val_c_loss /= len(validation_loader)
        val_wasserstein_distance /= len(validation_loader)
        
        logger.info(
            f"Validation - G_loss: {val_g_loss:.4f}, C_loss: {val_c_loss:.4f}, "
            f"W_distance: {val_wasserstein_distance:.4f}"
        )
        
        self.generator.train()
        self.critic.train()
        
        return {
            'val_g_loss': val_g_loss,
            'val_c_loss': val_c_loss,
            'val_wasserstein_distance': val_wasserstein_distance
        }
    
    def generate_samples(
        self,
        num_samples: int,
        noise: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Генерация синтетических образцов
        
        Args:
            num_samples: Количество образцов для генерации
            noise: Пользовательский шум (опционально)
            
        Returns:
            Сгенерированные данные
        """
        self.generator.eval()
        
        if noise is None:
            noise = torch.randn(num_samples, self.config.input_dim).to(self.device)
        
        with torch.no_grad():
            generated_data = self.generator(noise)
        
        return generated_data.cpu().numpy()
    
    def save_checkpoint(self, filepath: str):
        """Сохранение checkpoint модели"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'c_optimizer_state_dict': self.c_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"WGAN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Загрузка checkpoint модели"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"WGAN checkpoint loaded from {filepath}")


def main():
    """Пример использования WGAN"""
    
    # Конфигурация WGAN
    config = WGANConfig(
        sequence_length=60,
        output_dim=5,  # OHLCV
        batch_size=32,
        num_epochs=200,
        generator_lr=0.0001,
        critic_lr=0.0001,
        critic_iterations=5,
        use_gradient_penalty=True,
        gradient_penalty_weight=10.0,
        use_spectral_norm=False
    )
    
    # Создание синтетических данных для примера
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    
    # Генерация более сложных OHLCV данных
    prices = np.cumsum(np.random.randn(2000) * 0.01) + 100
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(2000) * 0.5),
        'low': prices - np.abs(np.random.randn(2000) * 0.5),
        'close': prices + np.random.randn(2000) * 0.2,
        'volume': np.random.lognormal(mean=10, sigma=1, size=2000)
    })
    
    # Создание dataset
    dataset = CryptoDataset(data, sequence_length=config.sequence_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Разделение на train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Обучение WGAN
    wgan = WGAN(config)
    training_history = wgan.train(train_dataloader, val_dataloader)
    
    # Генерация образцов
    samples = wgan.generate_samples(10)
    logger.info(f"Generated samples shape: {samples.shape}")
    
    # Сохранение модели
    wgan.save_checkpoint("models/wgan_final.pth")
    
    # Анализ качества
    logger.info("WGAN training metrics:")
    logger.info(f"Final Generator Loss: {training_history['g_loss'][-1]:.4f}")
    logger.info(f"Final Critic Loss: {training_history['c_loss'][-1]:.4f}")
    logger.info(f"Final Wasserstein Distance: {training_history['wasserstein_distance'][-1]:.4f}")


if __name__ == "__main__":
    main()