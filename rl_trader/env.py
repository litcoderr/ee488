import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_trader.data import load_price_data


@dataclass
class EnvConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    init_balance: float = 10_000.0
    max_shares_per_trade: int = 10
    window_size: int = 5
    action_mode: str = "discrete"  # "discrete" or "continuous"
    transaction_cost_pct: float = 0.001
    reward_scaling: float = 1.0
    seed: Optional[int] = None
    source: str = "yfinance"  # "yfinance", "csv", or "synthetic"
    data_dir: Optional[str] = None


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.tickers = config.tickers
        self.n_tickers = len(self.tickers)
        self.max_shares_per_trade = config.max_shares_per_trade
        self.window_size = max(1, config.window_size)
        self.transaction_cost_pct = config.transaction_cost_pct
        self.reward_scaling = config.reward_scaling
        self.rng = np.random.default_rng(config.seed)

        self.feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        self.data, self.dates = self._load_data()
        self.prices = self._extract_price_tensor()

        self.action_mode = config.action_mode
        if self.action_mode not in {"discrete", "continuous"}:
            raise ValueError("action_mode must be 'discrete' or 'continuous'")

        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(3 ** self.n_tickers)
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_tickers,),
                dtype=np.float32,
            )

        portfolio_low = np.zeros(2 + self.n_tickers, dtype=np.float32)
        portfolio_high = np.full(2 + self.n_tickers, np.finfo(np.float32).max)
        self.observation_space = spaces.Dict(
            {
                "portfolio": spaces.Box(
                    low=portfolio_low,
                    high=portfolio_high,
                    shape=portfolio_low.shape,
                    dtype=np.float32,
                ),
                "prices": spaces.Box(
                    low=-np.finfo(np.float32).max,
                    high=np.finfo(np.float32).max,
                    shape=(self.n_tickers * len(self.feature_columns) * self.window_size,),
                    dtype=np.float32,
                ),
            }
        )

        self._reset_state()

    def _load_data(self) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
        data = load_price_data(
            tickers=self.tickers,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            source=self.config.source,
            data_dir=self.config.data_dir,
            rng=self.rng,
        )
        dates = None
        for df in data.values():
            if dates is None:
                dates = df.index
            else:
                dates = dates.intersection(df.index)
        if dates is None or len(dates) < self.window_size + 1:
            raise ValueError("Not enough data to build the environment window")
        # Align all frames to common date index
        aligned = {t: df.loc[dates].reset_index(drop=True) for t, df in data.items()}
        return aligned, dates

    def _extract_price_tensor(self) -> np.ndarray:
        frames = []
        for t in self.tickers:
            df = self.data[t][self.feature_columns]
            frames.append(df.to_numpy(dtype=np.float32))
        stacked = np.stack(frames, axis=1)  # shape (T, n_tickers, features)
        return stacked

    def _reset_state(self):
        self.ptr = self.window_size - 1
        self.cash = float(self.config.init_balance)
        self.positions = np.zeros(self.n_tickers, dtype=np.float32)
        self.last_value = self._portfolio_value()
        self.done = False

    def _portfolio_value(self) -> float:
        prices = self.prices[self.ptr, :, self.feature_columns.index("Close")]
        return float(self.cash + np.dot(self.positions, prices))

    def _observe(self) -> Dict[str, np.ndarray]:
        start = self.ptr - self.window_size + 1
        window = self.prices[start : self.ptr + 1]  # shape (W, n_tickers, feat)
        price_obs = window.transpose(1, 0, 2).reshape(-1)
        portfolio_obs = np.concatenate(
            [
                np.array([self.cash], dtype=np.float32),
                self.positions.astype(np.float32),
                np.array([self._portfolio_value()], dtype=np.float32),
            ]
        )
        return {
            "portfolio": portfolio_obs.astype(np.float32),
            "prices": price_obs.astype(np.float32),
        }

    def _decode_discrete_action(self, action: int) -> np.ndarray:
        digits = []
        val = int(action)
        for _ in range(self.n_tickers):
            digits.append(val % 3)
            val //= 3
        digits.reverse()
        return np.array([d - 1 for d in digits], dtype=np.int64)

    def _apply_trade(self, action_vector: np.ndarray):
        prices = self.prices[self.ptr, :, self.feature_columns.index("Close")]
        for idx, act in enumerate(action_vector):
            direction = np.sign(act)
            if direction == 0:
                continue
            desired = int(min(self.max_shares_per_trade, abs(int(act))))
            if desired == 0:
                desired = self.max_shares_per_trade
            price = float(prices[idx])
            if direction > 0:
                affordable = int(self.cash // price)
                qty = min(desired, affordable)
            else:
                qty = min(desired, int(self.positions[idx]))
            if qty <= 0:
                continue
            trade_cost = qty * price
            fee = trade_cost * self.transaction_cost_pct
            if direction > 0:
                self.cash -= trade_cost + fee
                self.positions[idx] += qty
            else:
                self.cash += trade_cost - fee
                self.positions[idx] -= qty

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Call reset before stepping the environment")
        if self.action_mode == "discrete":
            action_vector = self._decode_discrete_action(int(action))
            scaled_actions = action_vector * self.max_shares_per_trade
        else:
            clipped = np.clip(action, -1.0, 1.0)
            scaled_actions = np.rint(clipped * self.max_shares_per_trade).astype(np.int64)
        self._apply_trade(scaled_actions)

        self.ptr += 1
        if self.ptr >= len(self.dates):
            self.ptr = len(self.dates) - 1
            self.done = True

        current_value = self._portfolio_value()
        reward = (current_value - self.last_value) * self.reward_scaling
        self.last_value = current_value

        obs = self._observe()
        terminated = self.ptr >= len(self.dates) - 1
        truncated = False
        info = {
            "portfolio_value": current_value,
            "date": self.dates[self.ptr],
            "cash": self.cash,
            "positions": self.positions.copy(),
        }
        self.done = terminated or truncated
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._observe(), {}

    def render(self):
        value = self._portfolio_value()
        date = self.dates[self.ptr]
        print(f"{date.date()} | Value: {value:.2f} | Cash: {self.cash:.2f} | Pos: {self.positions}")
