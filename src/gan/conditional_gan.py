"""
Conditional GAN Implementation for Crypto Trading Data

Advanced Conditional GAN (cGAN) for controlled synthesis of crypto trading data.
Enables generation conditioned on market regimes, volatility levels, and trading patterns.

Features:
- Market regime conditioning (bull/bear/sideways)
- Volatility level conditioning
- Trading volume conditioning
- Time-aware conditioning
- Multi-label conditional generation
- Enterprise monitoring and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from enum import Enum

from loguru import logger
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

from .vanilla_gan import CryptoDataset, GANConfig

# Configure logging
logger.add(
    "logs/conditional_gan_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


class MarketRegime(Enum):
    """Рыночные режимы для conditioning"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ConditionalGANConfig(GANConfig):
    """Конфигурация для Conditional GAN"""
    # Conditional parameters
    num_classes: int = 5  # Количество классов условий
    condition_dim: int = 10  # Размерность condition embedding
    use_label_smoothing: bool = True
    condition_types: List[str] = None
    
    # Market regime specific
    volatility_window: int = 20  # Окно для расчета волатильности
    trend_window: int = 10      # Окно для определения тренда
    
    # Advanced conditioning
    use_auxiliary_classifier: bool = True  # AC-GAN style
    auxiliary_weight: float = 1.0
    
    def __post_init__(self):
        super().__post_init__()
        if self.condition_types is None:
            self.condition_types = ['regime', 'volatility', 'volume', 'time_of_day']


class ConditionalCryptoDataset(Dataset):
    """Dataset с условными метками для cGAN"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: ConditionalGANConfig,
        sequence_length: int = 60,
        price_columns: List[str] = None,
        volume_column: str = 'volume'
    ):
        self.config = config
        self.sequence_length = sequence_length
        self.price_columns = price_columns or ['open', 'high', 'low', 'close']
        self.volume_column = volume_column
        
        self.data, self.conditions, self.scaler = self._preprocess_data(data)
        self.sequences, self.condition_sequences = self._create_sequences()
        
        logger.info(f"Conditional dataset created with {len(self.sequences)} sequences")
        logger.info(f"Condition types: {self.config.condition_types}")
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """Предобработка с созданием условных меток"""
        df = data.copy()
        
        # Базовая предобработка цен
        for col in self.price_columns:
            if col in df.columns:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        if self.volume_column in df.columns:
            df[f'{self.volume_column}_log'] = np.log1p(df[self.volume_column])
        
        # Создание условных меток
        conditions_df = pd.DataFrame(index=df.index)
        
        # 1. Market Regime (на основе ценовых движений)
        if 'regime' in self.config.condition_types:
            conditions_df['regime'] = self._compute_market_regime(df)
        
        # 2. Volatility Level
        if 'volatility' in self.config.condition_types:
            conditions_df['volatility_level'] = self._compute_volatility_level(df)
        
        # 3. Volume Level
        if 'volume' in self.config.condition_types:
            conditions_df['volume_level'] = self._compute_volume_level(df)
        
        # 4. Time of Day (циклическое кодирование)
        if 'time_of_day' in self.config.condition_types:
            if 'timestamp' in df.columns:
                conditions_df['hour_sin'], conditions_df['hour_cos'] = self._encode_time_cyclical(df)
        
        # 5. Price Level (кластеризация цен)
        if 'price_level' in self.config.condition_types:
            conditions_df['price_level'] = self._compute_price_level(df)
        
        # Удаляем NaN
        valid_idx = df.dropna().index
        df = df.loc[valid_idx]
        conditions_df = conditions_df.loc[valid_idx]
        
        # Нормализация основных данных
        feature_columns = [f'{col}_log_return' for col in self.price_columns]
        if f'{self.volume_column}_log' in df.columns:
            feature_columns.append(f'{self.volume_column}_log')
        
        scaler = StandardScaler()
        df_normalized = df[feature_columns].copy()
        df_normalized.loc[:, feature_columns] = scaler.fit_transform(df[feature_columns])
        
        return df_normalized, conditions_df, scaler
    
    def _compute_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Определение рыночного режима"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        close_prices = df['close']
        
        # Тренд (на основе скользящих средних)
        sma_short = close_prices.rolling(self.config.trend_window).mean()
        sma_long = close_prices.rolling(self.config.trend_window * 2).mean()
        
        # Волатильность
        returns = close_prices.pct_change()
        volatility = returns.rolling(self.config.volatility_window).std()
        vol_threshold = volatility.quantile(0.7)
        
        # Классификация режимов
        regime = pd.Series(0, index=df.index)  # Default: sideways
        
        # Bull market: upward trend + normal volatility
        bull_mask = (sma_short > sma_long * 1.01) & (volatility <= vol_threshold)
        regime.loc[bull_mask] = 1
        
        # Bear market: downward trend + normal volatility  
        bear_mask = (sma_short < sma_long * 0.99) & (volatility <= vol_threshold)
        regime.loc[bear_mask] = 2
        
        # High volatility: regardless of trend
        high_vol_mask = volatility > vol_threshold
        regime.loc[high_vol_mask] = 3
        
        return regime
    
    def _compute_volatility_level(self, df: pd.DataFrame) -> pd.Series:
        """Уровень волатильности"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        returns = df['close'].pct_change()
        volatility = returns.rolling(self.config.volatility_window).std()
        
        # Квантильная классификация
        vol_levels = pd.cut(
            volatility,
            bins=[-np.inf, volatility.quantile(0.33), volatility.quantile(0.67), np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        return vol_levels
    
    def _compute_volume_level(self, df: pd.DataFrame) -> pd.Series:
        """Уровень торгового объема"""
        if self.volume_column not in df.columns:
            return pd.Series(0, index=df.index)
        
        volume = df[self.volume_column]
        volume_ma = volume.rolling(20).mean()
        
        # Относительный объем
        relative_volume = volume / volume_ma
        
        # Классификация
        vol_levels = pd.cut(
            relative_volume,
            bins=[-np.inf, 0.7, 1.3, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        return vol_levels
    
    def _encode_time_cyclical(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Циклическое кодирование времени"""
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            hours = timestamps.dt.hour
        else:
            # Если нет timestamp, создаем искусственное время
            hours = pd.Series(range(len(df))) % 24
        
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        
        return hour_sin, hour_cos
    
    def _compute_price_level(self, df: pd.DataFrame) -> pd.Series:
        """Уровень цены через кластеризацию"""
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        prices = df['close'].values.reshape(-1, 1)
        
        # K-means кластеризация цен
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        price_clusters = kmeans.fit_predict(prices)
        
        return pd.Series(price_clusters, index=df.index)
    
    def _create_sequences(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Создание последовательностей с условными метками"""
        sequences = []
        condition_sequences = []
        
        data_values = self.data.values
        condition_values = self.conditions.values
        
        for i in range(len(data_values) - self.sequence_length + 1):
            # Данные последовательности
            sequence = data_values[i:i + self.sequence_length]
            sequences.append(sequence)
            
            # Условные метки (берем последнюю метку в последовательности)
            condition = condition_values[i + self.sequence_length - 1]
            condition_sequences.append(condition)
        
        return sequences, condition_sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        condition = torch.FloatTensor(self.condition_sequences[idx])
        return sequence, condition


class ConditionalGenerator(nn.Module):
    """Generator для Conditional GAN"""
    
    def __init__(self, config: ConditionalGANConfig):
        super(ConditionalGenerator, self).__init__()
        self.config = config
        
        # Embedding для условий
        self.condition_embedding = nn.Linear(
            config.num_classes, 
            config.condition_dim
        )
        
        # Основная архитектура
        layers = []
        
        # Входной слой (noise + condition)
        in_features = config.input_dim + config.condition_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Выходной слой
        layers.extend([
            nn.Linear(in_features, config.sequence_length * config.output_dim),
            nn.Tanh()
        ])
        
        self.main = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"Conditional Generator initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, noise, conditions):
        # Embedding условий
        condition_emb = self.condition_embedding(conditions)
        
        # Конкатенация noise и conditions
        input_combined = torch.cat([noise, condition_emb], dim=1)
        
        output = self.main(input_combined)
        return output.view(-1, self.config.sequence_length, self.config.output_dim)


class ConditionalDiscriminator(nn.Module):
    """Discriminator для Conditional GAN"""
    
    def __init__(self, config: ConditionalGANConfig):
        super(ConditionalDiscriminator, self).__init__()
        self.config = config
        
        # Embedding для условий
        self.condition_embedding = nn.Linear(
            config.num_classes,
            config.condition_dim
        )
        
        # Основная архитектура
        layers = []
        
        # Входной слой (data + condition)
        in_features = config.sequence_length * config.output_dim + config.condition_dim
        for hidden_dim in reversed(config.hidden_dims):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Основной выход (real/fake)
        self.discriminator_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        
        self.main = nn.Sequential(*layers)
        
        # Auxiliary classifier (если используется AC-GAN)
        if config.use_auxiliary_classifier:
            self.auxiliary_head = nn.Sequential(
                nn.Linear(in_features, config.num_classes),
                nn.Softmax(dim=1)
            )
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"Conditional Discriminator initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_data, conditions):
        # Flatten input data
        input_flat = input_data.view(-1, self.config.sequence_length * self.config.output_dim)
        
        # Embedding условий
        condition_emb = self.condition_embedding(conditions)
        
        # Конкатенация data и conditions
        combined_input = torch.cat([input_flat, condition_emb], dim=1)
        
        # Основные features
        features = self.main(combined_input)
        
        # Real/fake classification
        discriminator_output = self.discriminator_head(features)
        
        # Auxiliary classification (если используется)
        auxiliary_output = None
        if self.config.use_auxiliary_classifier:
            auxiliary_output = self.auxiliary_head(features)
        
        return discriminator_output, auxiliary_output


class ConditionalGAN:
    """
    Conditional GAN для контролируемой генерации криптовалютных данных
    
    Возможности:
    - Генерация по рыночным режимам
    - Контроль волатильности и объема
    - Временное conditioning
    - AC-GAN стиль auxiliary classification
    """
    
    def __init__(self, config: ConditionalGANConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Инициализация моделей
        self.generator = ConditionalGenerator(config).to(self.device)
        self.discriminator = ConditionalDiscriminator(config).to(self.device)
        
        # Оптимизаторы
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss functions
        self.adversarial_criterion = nn.BCELoss()
        if config.use_auxiliary_classifier:
            self.auxiliary_criterion = nn.CrossEntropyLoss()
        
        # Метрики
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'g_auxiliary_loss': [],
            'd_auxiliary_loss': [],
            'epoch': [],
            'timestamp': []
        }
        
        logger.info(f"Conditional GAN initialized on {self.device}")
        logger.info(f"Auxiliary classifier: {config.use_auxiliary_classifier}")
    
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        
        return torch.device(device)
    
    def train(
        self,
        dataloader: DataLoader,
        validation_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """Обучение Conditional GAN"""
        logger.info(f"Starting Conditional GAN training for {self.config.num_epochs} epochs")
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_g_aux_loss = 0.0
            epoch_d_aux_loss = 0.0
            
            for batch_idx, (real_data, real_conditions) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                real_conditions = real_conditions.to(self.device)
                batch_size = real_data.size(0)
                
                # Обучение Discriminator
                d_loss, d_aux_loss = self._train_discriminator(real_data, real_conditions, batch_size)
                epoch_d_loss += d_loss
                epoch_d_aux_loss += d_aux_loss
                
                # Обучение Generator
                g_loss, g_aux_loss = self._train_generator(real_conditions, batch_size)
                epoch_g_loss += g_loss
                epoch_g_aux_loss += g_aux_loss
                
                # Логирование
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f} "
                        f"D_aux: {d_aux_loss:.4f} G_aux: {g_aux_loss:.4f}"
                    )
            
            # Усреднение loss за эпоху
            num_batches = len(dataloader)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_aux_loss = epoch_g_aux_loss / num_batches
            avg_d_aux_loss = epoch_d_aux_loss / num_batches
            
            # Сохранение метрик
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['d_loss'].append(avg_d_loss)
            self.training_history['g_auxiliary_loss'].append(avg_g_aux_loss)
            self.training_history['d_auxiliary_loss'].append(avg_d_aux_loss)
            self.training_history['epoch'].append(epoch)
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Валидация
            if validation_loader and epoch % self.config.validate_interval == 0:
                self._validate(validation_loader)
            
            # Сохранение модели
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"conditional_gan_checkpoint_epoch_{epoch}.pth")
            
            logger.info(
                f"Epoch [{epoch}/{self.config.num_epochs}] completed - "
                f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, "
                f"G_aux: {avg_g_aux_loss:.4f}, D_aux: {avg_d_aux_loss:.4f}"
            )
        
        logger.info("Conditional GAN training completed successfully")
        return self.training_history
    
    def _train_discriminator(
        self, 
        real_data: torch.Tensor, 
        real_conditions: torch.Tensor, 
        batch_size: int
    ) -> Tuple[float, float]:
        """Обучение дискриминатора"""
        self.d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        if self.config.use_label_smoothing:
            real_labels -= self.config.label_smoothing * torch.rand_like(real_labels)
        
        real_output, real_aux_output = self.discriminator(real_data, real_conditions)
        real_loss = self.adversarial_criterion(real_output, real_labels)
        
        # Auxiliary loss для real data
        real_aux_loss = 0.0
        if self.config.use_auxiliary_classifier and real_aux_output is not None:
            # Преобразуем conditions в class labels
            real_class_labels = torch.argmax(real_conditions, dim=1)
            real_aux_loss = self.auxiliary_criterion(real_aux_output, real_class_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_conditions = self._sample_conditions(batch_size)  # Случайные условия
        fake_data = self.generator(noise, fake_conditions)
        
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output, fake_aux_output = self.discriminator(fake_data.detach(), fake_conditions)
        fake_loss = self.adversarial_criterion(fake_output, fake_labels)
        
        # Auxiliary loss для fake data
        fake_aux_loss = 0.0
        if self.config.use_auxiliary_classifier and fake_aux_output is not None:
            fake_class_labels = torch.argmax(fake_conditions, dim=1)
            fake_aux_loss = self.auxiliary_criterion(fake_aux_output, fake_class_labels)
        
        # Общий loss
        total_aux_loss = real_aux_loss + fake_aux_loss
        d_loss = real_loss + fake_loss + self.config.auxiliary_weight * total_aux_loss
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item(), total_aux_loss if isinstance(total_aux_loss, float) else total_aux_loss.item()
    
    def _train_generator(self, real_conditions: torch.Tensor, batch_size: int) -> Tuple[float, float]:
        """Обучение генератора"""
        self.g_optimizer.zero_grad()
        
        # Генерация fake data с теми же условиями
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_data = self.generator(noise, real_conditions)
        
        # Пытаемся обмануть дискриминатор
        fake_labels = torch.ones(batch_size, 1).to(self.device)
        fake_output, fake_aux_output = self.discriminator(fake_data, real_conditions)
        
        g_loss = self.adversarial_criterion(fake_output, fake_labels)
        
        # Auxiliary loss
        g_aux_loss = 0.0
        if self.config.use_auxiliary_classifier and fake_aux_output is not None:
            real_class_labels = torch.argmax(real_conditions, dim=1)
            g_aux_loss = self.auxiliary_criterion(fake_aux_output, real_class_labels)
            
            total_g_loss = g_loss + self.config.auxiliary_weight * g_aux_loss
        else:
            total_g_loss = g_loss
        
        total_g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), g_aux_loss if isinstance(g_aux_loss, float) else g_aux_loss.item()
    
    def _sample_conditions(self, batch_size: int) -> torch.Tensor:
        """Генерация случайных условий"""
        # Создаем one-hot кодированные условия
        conditions = torch.zeros(batch_size, self.config.num_classes).to(self.device)
        
        # Случайный выбор класса для каждого образца
        random_classes = torch.randint(0, self.config.num_classes, (batch_size,))
        conditions.scatter_(1, random_classes.unsqueeze(1).to(self.device), 1)
        
        return conditions
    
    def _validate(self, validation_loader: DataLoader):
        """Валидация модели"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        
        with torch.no_grad():
            for real_data, real_conditions in validation_loader:
                real_data = real_data.to(self.device)
                real_conditions = real_conditions.to(self.device)
                batch_size = real_data.size(0)
                
                # Generator loss
                noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
                fake_data = self.generator(noise, real_conditions)
                fake_output, _ = self.discriminator(fake_data, real_conditions)
                fake_labels = torch.ones(batch_size, 1).to(self.device)
                g_loss = self.adversarial_criterion(fake_output, fake_labels)
                val_g_loss += g_loss.item()
                
                # Discriminator loss
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output, _ = self.discriminator(real_data, real_conditions)
                real_loss = self.adversarial_criterion(real_output, real_labels)
                
                fake_labels_d = torch.zeros(batch_size, 1).to(self.device)
                fake_output_d, _ = self.discriminator(fake_data, real_conditions)
                fake_loss = self.adversarial_criterion(fake_output_d, fake_labels_d)
                
                d_loss = real_loss + fake_loss
                val_d_loss += d_loss.item()
        
        val_g_loss /= len(validation_loader)
        val_d_loss /= len(validation_loader)
        
        logger.info(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")
        
        self.generator.train()
        self.discriminator.train()
    
    def generate_samples(
        self,
        num_samples: int,
        conditions: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Генерация с заданными условиями
        
        Args:
            num_samples: Количество образцов
            conditions: Условия (если None, генерируются случайно)
            noise: Шум (если None, генерируется случайно)
        """
        self.generator.eval()
        
        if noise is None:
            noise = torch.randn(num_samples, self.config.input_dim).to(self.device)
        
        if conditions is None:
            conditions = self._sample_conditions(num_samples)
        else:
            conditions = conditions.to(self.device)
        
        with torch.no_grad():
            generated_data = self.generator(noise, conditions)
        
        return generated_data.cpu().numpy()
    
    def generate_by_regime(
        self,
        num_samples: int,
        regime: MarketRegime,
        **kwargs
    ) -> np.ndarray:
        """Генерация для специфичного рыночного режима"""
        # Создаем условия для указанного режима
        conditions = torch.zeros(num_samples, self.config.num_classes).to(self.device)
        
        # Маппинг режимов на индексы
        regime_mapping = {
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.BULL: 1,
            MarketRegime.BEAR: 2,
            MarketRegime.HIGH_VOLATILITY: 3,
            MarketRegime.LOW_VOLATILITY: 4
        }
        
        regime_idx = regime_mapping.get(regime, 0)
        conditions[:, regime_idx] = 1.0
        
        return self.generate_samples(num_samples, conditions, **kwargs)
    
    def save_checkpoint(self, filepath: str):
        """Сохранение checkpoint модели"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Conditional GAN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Загрузка checkpoint модели"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Conditional GAN checkpoint loaded from {filepath}")


def main():
    """Пример использования Conditional GAN"""
    
    # Конфигурация cGAN
    config = ConditionalGANConfig(
        sequence_length=60,
        output_dim=5,  # OHLCV
        num_classes=5,  # Количество условных классов
        condition_dim=10,
        batch_size=32,
        num_epochs=150,
        learning_rate=0.0002,
        use_auxiliary_classifier=True,
        auxiliary_weight=1.0,
        condition_types=['regime', 'volatility', 'volume']
    )
    
    # Создание синтетических данных
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=3000, freq='1H')
    
    # Симуляция различных рыночных режимов
    base_price = 1000
    prices = []
    volumes = []
    
    for i in range(3000):
        # Создаем разные режимы
        if i < 1000:  # Bull market
            price_change = np.random.normal(0.001, 0.01)
            volume_mult = np.random.lognormal(0, 0.3)
        elif i < 2000:  # Bear market  
            price_change = np.random.normal(-0.001, 0.01)
            volume_mult = np.random.lognormal(0, 0.5)
        else:  # High volatility
            price_change = np.random.normal(0, 0.03)
            volume_mult = np.random.lognormal(0, 0.8)
        
        base_price *= (1 + price_change)
        prices.append(base_price)
        volumes.append(1000000 * volume_mult)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.randn() * p * 0.005) for p in prices],
        'low': [p - abs(np.random.randn() * p * 0.005) for p in prices],
        'close': [p + np.random.randn() * p * 0.002 for p in prices],
        'volume': volumes
    })
    
    # Создание conditional dataset
    dataset = ConditionalCryptoDataset(data, config, sequence_length=config.sequence_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Разделение на train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Обучение Conditional GAN
    cgan = ConditionalGAN(config)
    training_history = cgan.train(train_dataloader, val_dataloader)
    
    # Генерация по различным режимам
    bull_samples = cgan.generate_by_regime(5, MarketRegime.BULL)
    bear_samples = cgan.generate_by_regime(5, MarketRegime.BEAR)
    high_vol_samples = cgan.generate_by_regime(5, MarketRegime.HIGH_VOLATILITY)
    
    logger.info(f"Bull market samples shape: {bull_samples.shape}")
    logger.info(f"Bear market samples shape: {bear_samples.shape}")
    logger.info(f"High volatility samples shape: {high_vol_samples.shape}")
    
    # Сохранение модели
    cgan.save_checkpoint("models/conditional_gan_final.pth")
    
    logger.info("Conditional GAN training completed successfully!")


if __name__ == "__main__":
    main()