from typing import Any, Callable, Iterable, List, Optional, Type, Union, Mapping, Tuple
from commonroad.scenario.scenario import Scenario
from commonroad_geometric_io.dataset.collection import BaseDatasetCollector
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_geometric_io.common.io import find_scenario_files
from commonroad_geometric_io.common.progress import BaseProgressReporter, ProgressReporter
from commonroad_geometric_io.modeling.utils import to_float32
from torch_geometric.data import HeteroData
from torch_geometric.data.data import Data
import os
import torch
import shutil
import random

class TrafficDataCollector(BaseDatasetCollector):

    def export_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_total_samples: int = -1,
        max_samples_per_scenario: int = -1,
        max_total_scenarios: int = -1,
        overwrite: bool = True,
        shuffle_scenarios: bool = False,
        skip_subvariants: bool = False,
        callback_fn: Optional[Callable[[Scenario], None]] = None,
        dataset_size_percent: Tuple[int, int] = (80, 20),
        **collect_kwargs
    ):

        training_set_size_percent: int = dataset_size_percent[0]
        test_set_size_percent: int = dataset_size_percent[1]

        assert training_set_size_percent >= 0, "Negative percentage for training set size not allowed"
        assert test_set_size_percent >= 0, "Negative percentage for test set size not allowed"
        assert training_set_size_percent + test_set_size_percent == 100, "Dataset sizes don't add up to 100"

        scenario_files = find_scenario_files(
            input_dir,
            shuffle=shuffle_scenarios,
            max_results=max_total_scenarios,
            skip_subvariants=skip_subvariants
        )

        if shuffle_scenarios:
            random.shuffle(scenario_files)

        assert len(scenario_files) > 0, f"Found no scenario files in '{input_dir}'"
        print(f"Found {len(scenario_files)} scenario files")

        num_scenarios = len(scenario_files)
        training_scenarios = scenario_files[:int(num_scenarios * training_set_size_percent / 100)]
        test_scenarios = scenario_files[int(num_scenarios * training_set_size_percent / 100):]

        if overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        self.collect_split(training_scenarios, max_samples_per_scenario, max_total_samples, output_dir, "training")
        self.collect_split(test_scenarios, max_samples_per_scenario, max_total_samples, output_dir, "test")

    def collect_split(self, scenario_files, max_samples_per_scenario, max_total_samples, output_dir, name):
        all_samples = []
        for i, samples in enumerate(self.collect(
            scenarios=scenario_files,
            max_total_samples=max_total_samples,
            max_samples_per_scenario=max_samples_per_scenario,
        )):
            all_samples.extend(samples)

        file_path = os.path.join(output_dir, f"{name}.pth")
        print(f"Saving {len(all_samples)} samples to {file_path}")
        torch.save(all_samples, file_path)
        return

    def collect(
            self,
            scenarios: List[Union[Scenario, str]],
            max_total_samples: int = 0,
            max_samples_per_scenario: int = 0,
            callback_fn: Optional[Callable[[Scenario], None]] = None,
            scenario_post_processors: Optional[List[Callable[..., Union[List[Data], List[HeteroData]]]]] = None,
            progress: Type[BaseProgressReporter] = ProgressReporter,
    ) -> Union[Iterable[List[Data]], Iterable[List[HeteroData]]]:
        """Extracts graphs from each of the given CommonRoad scenarios.

        Args:
            scenarios (List[Scenario | str]):
                Iterable sequence of CommonRoad scenarios (or paths to scenarios).
            max_total_samples (int, optional):
                Maximum number of samples to collect. Defaults to 0 meaning unlimited samples.
            max_samples_per_scenario (int, optional):
                Maximum number of samples to collect per scenario. Defaults to 0 meaning unlimited samples.
            callback_fn (callable, optional):
                Optional callback function executed for each scenario, e.g. for plotting.
            scenario_post_processors (list of callables, optional):
                List of scenario post-processing functions. They are called in order, for each of the
                processed scenarios.
            progress (type of BaseProgressReporter, optional):
                Progress reporter class. The default does not print progress.
        Returns:
            Iterable[List[Data]] | Iterable[List[HeteroData]]:
                A list of extracted and post-processed samples for each CommonRoad scenario.
        """
        if max_total_samples < 1:
            max_total_samples = float("inf")
        if max_samples_per_scenario < 1:
            max_samples_per_scenario = float("inf")
        if scenario_post_processors is None:
            scenario_post_processors = []

        progress = progress(name="Scenarios", total=len(scenarios))

        total_sample_counter = 0
        aggr_total_extractor_lengths = 0
        exceptions = []
        for scenario_idx, scenario in enumerate(scenarios):
            progress.update(scenario_idx)

            if isinstance(scenario, str):
                scenario, _ = CommonRoadFileReader(scenario).open()

            if callback_fn is not None:
                callback_fn(scenario)

            extractor = self._extractor_cls(scenario, **self._extractor_kwargs)
            if len(extractor) == 0:
                continue

            progress_scenario = progress.nested_progress()("Samples", total=len(extractor))
            post_processing_progress = progress_scenario.nested_progress()

            scenario_sample_buffer = []
            aggr_total_extractor_lengths += len(extractor)

            step = int(len(extractor) / max_samples_per_scenario)
            if step == 0:
                step = 1
            for t in range(0, len(extractor), step):
                raw_data = extractor.extract(t)
                data = to_float32(raw_data)
                scenario_sample_buffer.append(data)
                total_sample_counter += 1

                progress_scenario.update(t)
                progress.set_postfix_str(f">= {total_sample_counter} total samples", refresh=False)

                if total_sample_counter >= max_total_samples or \
                        len(scenario_sample_buffer) >= max_samples_per_scenario:
                    break

            extractor.close()
            progress_scenario.close()

            for post_processor in scenario_post_processors:
                scenario_sample_buffer = post_processor(scenario_sample_buffer, progress=post_processing_progress)
            yield scenario_sample_buffer

            if total_sample_counter >= max_total_samples:
                break

        progress.close()
        print(f"Collected {total_sample_counter}/{aggr_total_extractor_lengths} samples from {len(scenarios)} scenarios with {len(exceptions)} errors:")
        print(exceptions)
