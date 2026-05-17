"""
nanoLLM/src/training/runner.py

Orchestrates data loading, processing, and training run.
This module can be used by all types of entry points
(e.g., CLI scripts, notebooks) because it consumes
agnostic, typed configurations.
"""

import dataclasses
import logging
from pathlib import Path

from src.checkpoint import CheckpointMetadata, save_checkpoint
from src.config import TokenizerConfig, TrainingConfig
from src.data.io import load_text_from_file
from src.data.processor import Processor
from src.model.model import NanoLLM
from src.training.schema import MetricsHistory, ResumeContext
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates a single training run end-to-end:
    loads data, preprocesses it, trains the model, and
    optionally persists a new checkpoint.

    Also manages loading weights from a checkpoint via the
    optional resume context.
    """

    def __init__(
        self,
        *,
        model: NanoLLM,
        tokenizer_config: TokenizerConfig,
        data_source: Path,
        training_config: TrainingConfig,
        checkpoint_destination: Path | None,
        resume_from: ResumeContext | None = None,
    ) -> None:
        """
        Args:
            For execution of training...
            - model: the model to train
            - tokenizer_config: tokenizer settings used for data preprocessing
            - data_source: path to the raw text file
            - training_config: training parameters

            For checkpoint loading and persistence...
            - checkpoint_destination: where to write the checkpoint after training;
                                if None, checkpoint is not persisted
            - resume_from: ResumeContext with the data required when resuming a prior run;
                                if None, checkpoint (pre-trained) weights is not loaded
        """
        self.model = model
        self.tokenizer_config = tokenizer_config
        self.data_source = data_source
        self.training_config = training_config
        self.checkpoint_destination = checkpoint_destination
        self.resume_from = resume_from

    def run(self) -> None:
        """Shared boilerplate for training pathways:
        loads data, preprocesses it, trains the model.

        It also manages loading weights from a checkpoint
        and persisting a new checkpoint after training run.

        Raises:
            FileNotFoundError: if data_source does not exist
            ValueError: if the dataset is empty or yields no complete batches
        """

        # Log checkpoint characteristics
        if self.checkpoint_destination is None:
            logger.warning("No checkpoint_destination path provided so no checkpoint will be persisted.")
        if self.resume_from is None:
            logger.info("No resume context received so no pre-trained weights will be loaded.")

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
        previous_epochs_completed, checkpoint_source = self._unpack_resume_context()
        header_lines = self._build_training_header_lines(
            checkpoint_source=checkpoint_source,
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

    # --- private methods ---

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
        checkpoint_source: Path | None,
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
            f"\tcheckpoint source:       {checkpoint_source}",
            f"\tprevious epochs trained: {previous_epochs}",
            "",
            f"\tcheckpoint destination: {self.checkpoint_destination}",
        ]

    def _unpack_resume_context(self) -> tuple[int, Path | None]:
        """Return (previous_epochs_completed, checkpoint_source), defaulting to (0, None)
        when no resume context is provided."""
        if self.resume_from is None:
            return 0, None
        return self.resume_from.previous_epochs_completed, self.resume_from.source

    def _persist_checkpoint(
        self,
        *,
        metrics_history: MetricsHistory,
        cumulative_epochs_completed: int,
    ) -> None:
        """Assemble checkpoint metadata and persist the model.

        Skips persistence (logging only) when destination is None.
        """
        if self.checkpoint_destination is None:
            logger.info("Checkpoint path undefined. No checkpoint persisted.")
            return

        final_loss = metrics_history.final_train_loss
        metadata = CheckpointMetadata(
            cumulative_epochs_completed=cumulative_epochs_completed,
            final_loss=final_loss,
            model_config=dataclasses.asdict(self.model.config),
            training_config=dataclasses.asdict(self.training_config),
            tokenizer_config=dataclasses.asdict(self.tokenizer_config),
        )
        save_checkpoint(self.model, self.checkpoint_destination, metadata=metadata)
