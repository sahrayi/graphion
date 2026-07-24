"""
Autoencoder dimensionality reduction.

Uses PyTorch implementation internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

from graphora.core.types import TId

from .base_reducer import BaseReducer


class Autoencoder(
    BaseReducer[
        TId,
        Sequence[float],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    Neural network based dimensionality reduction.

    Learns a nonlinear latent representation
    using an autoencoder architecture.

    Requires:

        pip install torch

    Suitable for:

    - dense embeddings
    - nonlinear feature spaces
    - learned representations
    """

    def __init__(
        self,
        *,
        output_dimension: int = 32,
        hidden_dimension: int = 128,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int | None = 42,
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
        )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be greater than zero."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        self.hidden_dimension = hidden_dimension
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._load_dependencies()

    def _load_dependencies(
        self,
    ) -> None:
        """
        Lazy load PyTorch dependencies.
        """

        try:
            import torch
            import torch.nn as nn

        except ImportError as exc:

            raise ImportError(
                "Autoencoder reducer requires "
                "'torch'. "
                "Install with: pip install torch"
            ) from exc

        self._torch = torch
        self._nn = nn

    def reduce_features(
        self,
        features: tuple[Sequence[float], ...],
    ) -> tuple[tuple[float, ...], ...]:
        """
        Train autoencoder and extract latent vectors.
        """

        if not features:
            return ()

        torch = self._torch
        nn = self._nn

        if self.random_state is not None:

            torch.manual_seed(
                self.random_state,
            )

        data = torch.tensor(
            features,
            dtype=torch.float32,
        )

        input_dimension = data.shape[1]

        encoder = nn.Sequential(
            nn.Linear(
                input_dimension,
                self.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dimension,
                self.output_dimension,
            ),
        )

        decoder = nn.Sequential(
            nn.Linear(
                self.output_dimension,
                self.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dimension,
                input_dimension,
            ),
        )

        class Model(
            nn.Module,
        ):
            def __init__(
                self,
            ):
                super().__init__()

                self.encoder = encoder
                self.decoder = decoder

            def forward(
                self,
                x,
            ):
                latent = self.encoder(
                    x,
                )

                reconstructed = self.decoder(
                    latent,
                )

                return reconstructed

        model = Model()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.MSELoss()

        model.train()

        dataset = torch.utils.data.TensorDataset(
            data,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(
            self.epochs,
        ):

            for (
                batch,
            ) in loader:

                x = batch[0]

                optimizer.zero_grad()

                reconstructed = model(x)

                loss = loss_function(
                    reconstructed,
                    x,
                )

                loss.backward()

                optimizer.step()

        model.eval()

        with torch.no_grad():

            latent = model.encoder(
                data,
            )

        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in latent.numpy()
        )