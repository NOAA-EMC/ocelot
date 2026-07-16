
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig


class OcelotFactory:
    @staticmethod
    def build_kwargs(
        model_config: ModelConfig,
        training_config: TrainingConfig,
        observation_config: dict,
        feature_stats: dict,
        instrument_weights: dict,
        channel_weights: dict,
        mesh_variable_config: dict | None = None,
        verbose: bool = False,
    ) -> dict:
        validation_csv = training_config.validation.csv
        processor_config = model_config.processor
        if hasattr(processor_config, 'window'):
            expected_window = (
                training_config.data.window_hours
                // training_config.data.latent_step_hours
            )
            if processor_config.window != expected_window:
                raise ValueError(
                    f"Processor window ({processor_config.window}) must equal "
                    f"data.window_hours / data.latent_step_hours ({expected_window})"
                )

        return dict(
            observation_config=observation_config,
            model_config=model_config.to_dict(),
            optimizer_config={
                'type': training_config.optimizer.type,
                **training_config.optimizer.to_dict(),
            },
            loss_config={
                'type': training_config.loss.type,
                **training_config.loss.to_dict(),
            },
            schedule_config={
                'type': training_config.schedule.type,
                **training_config.schedule.to_dict(),
            },
            mesh_variable_config=mesh_variable_config,
            feature_stats=feature_stats,
            instrument_weights=instrument_weights,
            channel_weights=channel_weights,
            max_rollout_steps=training_config.data.max_rollout_steps,
            latent_step_hours=training_config.data.latent_step_hours,
            val_csv_enabled=not validation_csv.disable,
            val_csv_out_dir=validation_csv.output_dir,
            val_csv_num_batches=validation_csv.num_batches,
            val_csv_every_n_epochs=validation_csv.every_n_epochs,
            val_csv_max_rows=validation_csv.max_rows,
            val_csv_sample_seed=validation_csv.seed,
            detect_anomaly=training_config.debug_mode,
            verbose=verbose or training_config.verbose,
        )

    @staticmethod
    def create_model(**kwargs):
        from .ocelot import Ocelot

        return Ocelot(**OcelotFactory.build_kwargs(**kwargs))
        