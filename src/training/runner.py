"""
nanoLLM/src/training/runner.py

Orchestrates data loading, processing, and training run.
This module can be used by all types of entry points
(e.g., CLI scripts, notebooks) because it consumes
agnostic, typed configurations.
"""

import logging
from pathlib import Path

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.data.io import load_text_from_file
from src.data.processor import Processor
from src.model.model import NanoLLM, count_params
from src.training.checkpoint import build_and_save_checkpoint, restore_from_checkpoint
from src.training.schema import CheckpointMetadata, MetricsHistory
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates a single training run end-to-end:
    builds (or restores) the model, loads data, preprocesses it,
    trains the model, and optionally persists a new checkpoint.

    Callers supply either:
        - model_config + tokenizer_config (fresh training), or
        - checkpoint_source (resume training; configs are read from the bundle)

    If both are supplied, the checkpoint's configs supersede the explicit
    arguments and a warning is logged.
    """

    def __init__(
        self,
        *,
        data_source: Path,
        training_config: TrainingConfig,
        checkpoint_destination: Path | None,
        model_config: ModelConfig | None = None,
        tokenizer_config: TokenizerConfig | None = None,
        checkpoint_source: Path | None = None,
    ) -> None:
        """
        Args:
            For execution of training...
            - data_source: path to the raw text file
            - training_config: training parameters
            - checkpoint_destination: where to write the checkpoint after training;
                                if None, checkpoint is not persisted

            For model construction (one of)...
            - model_config + tokenizer_config: build a fresh, untrained model
            - checkpoint_source: load a pre-trained model and its configs from
                                checkpoint bundle. Supersedes model_config/tokenizer_config

        Raises:
            ValueError: if neither (model_config + tokenizer_config) nor
                        checkpoint_source is provided.
        """
        if checkpoint_source is None and (model_config is None or tokenizer_config is None):
            raise ValueError(
                "Runner requires either checkpoint_source (to load pre-trained model) or both ",
                "model_config and tokenizer_config (to load fresh untrained model)."
            )

        self.data_source = data_source
        self.training_config = training_config
        self.checkpoint_destination = checkpoint_destination
        self.checkpoint_source = checkpoint_source
        self._initial_model_config = model_config
        self._initial_tokenizer_config = tokenizer_config

        # Populated by _prepare_model() during run().
        self.model: NanoLLM | None = None
        self.tokenizer_config: TokenizerConfig | None = None
        self.previous_metadata: CheckpointMetadata | None = None

    def run(self) -> MetricsHistory:
        """Shared boilerplate for training pathways:
        builds (or restores) the model, loads data, preprocesses it,
        trains the model, and optionally persists a new checkpoint.

        Raises:
            FileNotFoundError: if data_source does not exist
            ValueError: if the dataset is empty or yields no complete batches
        """

        # Log checkpoint characteristics
        if self.checkpoint_destination is None:
            logger.warning("No checkpoint_destination path provided so no checkpoint will be persisted.")

        # Build or restore the model
        self._prepare_model()

        # Load the data
        logger.info("Loading data ...")
        stories = load_text_from_file(
            file_path=self.data_source,
            delimiter=self.tokenizer_config.delimiter,
            max_paragraphs=self.training_config.max_stories,
        )
        logger.info("Data loading complete.")

        # Validate data characteristics; Cross-domain considerations (not strictly loading or training)
        record_count = len(stories)
        if record_count == 0:
            raise ValueError("Dataset is empty. Training aborted.")
        batches_per_epoch = self._calculate_batches(record_count)

        # Preprocess the data
        logger.info("Processing data ...")
        data_processor = Processor(
            model_config=self.model.config,
            tokenizer_config=self.tokenizer_config,
            training_config=self.training_config,
        )
        dataloader = data_processor.process(stories)
        logger.info("Data processing complete.")

        # Log training configuration
        previous_epochs_completed = self._previous_epochs_completed()
        header_lines = self._build_training_header_lines(
            previous_epochs=previous_epochs_completed,
        )
        logger.info("\n\n%s\n\n", "\n".join(header_lines))

        # Train the model
        logger.info(self._format_banner("Commencing training..."))
        trainer = Trainer(
            model=self.model,
            dataloader=dataloader,
            batches_per_epoch=batches_per_epoch,
            training_config=self.training_config,
        )
        metrics_history = trainer.train()
        logger.info(self._format_banner("Training complete."))

        # Persist checkpoint
        cumulative_epochs_completed = previous_epochs_completed + self.training_config.epochs
        logger.info(
            f"\n\tAll {self.training_config.epochs} epochs completed.\n"
            f"\tThis is in addition to {previous_epochs_completed} epochs accumulated during previous trainings.\n"
            f"\tCumulative epochs completed: {cumulative_epochs_completed}.\n"
        )
        self._persist_checkpoint(
            metrics_history=metrics_history,
            cumulative_epochs_completed=cumulative_epochs_completed,
        )

        return metrics_history

    # --- private methods ---

    def _prepare_model(self) -> None:
        """Populate self.model, self.tokenizer_config, and self.previous_metadata.

        If checkpoint_source is set, restore from it (and warn if explicit
        configs were also passed, since they will be ignored).
        Otherwise, construct a fresh NanoLLM from the provided model_config
        and use the provided tokenizer_config.
        """
        if self.checkpoint_source is not None:
            if self._initial_model_config is not None or self._initial_tokenizer_config is not None:
                logger.warning(
                    "checkpoint_source provided; model_config and tokenizer_config "
                    "arguments will be ignored in favor of values from the checkpoint."
                )
            logger.info(f"Loading checkpoint from {self.checkpoint_source}")
            self.model, self.tokenizer_config, self.previous_metadata = restore_from_checkpoint(
                self.checkpoint_source
            )
        else:
            logger.info("No checkpoint_source provided; building fresh model from model_config.")
            self.model = NanoLLM(self._initial_model_config)
            self.tokenizer_config = self._initial_tokenizer_config
            self.previous_metadata = None
        logger.info(f"Model ready ({count_params(self.model)} parameters)")

    def _calculate_batches(self, record_count: int) -> int:
        """Compute and validate batches per epoch from dataset size and batch size.

        Raises:
            ValueError: if batches_per_epoch <= 0
        """
        batches_per_epoch = record_count // self.training_config.batch_size
        if batches_per_epoch <= 0:
            raise ValueError(
                f"Calculated {batches_per_epoch} batches per epoch but must be > 0. Training aborted."
            )
        logger.info(f"Calculated batches per epoch: {batches_per_epoch}")
        return batches_per_epoch

    @staticmethod
    def _format_banner(text: str, width: int = 30) -> str:
        """Wrap a short message in dashed top/bottom banner lines with breathing room."""
        edge = "-" * width
        return f"\n\n{edge}\n{text}\n{edge}\n\n"

    def _build_training_header_lines(
        self,
        *,
        previous_epochs: int,
    ) -> list[str]:
        """Build a per-invocation summary header as a list of lines."""
        return [
            f"\tepochs (this run):      {self.training_config.epochs}",
            f"\tdata source:            {self.data_source}",
            f"\tmax stories:            {self.training_config.max_stories}",
            f"\tbatch size:             {self.training_config.batch_size}",
            f"\tshuffle:                {self.training_config.shuffle}",
            f"\tseed:                   {self.training_config.seed}",
            "",
            f"\tprevious epochs trained: {previous_epochs}",
            "",
            f"\tcheckpoint destination: {self.checkpoint_destination}",
        ]

    def _previous_epochs_completed(self) -> int:
        """Cumulative epochs from prior checkpoint, defaulting to 0 when no metadata is provided."""
        if self.previous_metadata is None:
            return 0
        return self.previous_metadata.cumulative_epochs_completed

    def _persist_checkpoint(
        self,
        *,
        metrics_history: MetricsHistory,
        cumulative_epochs_completed: int,
    ) -> None:
        """Persist the model and its metadata to the checkpoint destination.

        Skips persistence (logging only) when destination is None.
        """
        if self.checkpoint_destination is None:
            logger.info("Checkpoint path undefined. No checkpoint persisted.")
            return

        build_and_save_checkpoint(
            self.model,
            self.checkpoint_destination,
            training_config=self.training_config,
            tokenizer_config=self.tokenizer_config,
            cumulative_epochs_completed=cumulative_epochs_completed,
            final_loss=metrics_history.final_train_loss,
        )
