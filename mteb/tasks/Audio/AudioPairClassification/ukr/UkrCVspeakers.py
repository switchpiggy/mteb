from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata
import random
import datasets
import pandas as pd
import numpy as np
from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)

random.seed(42)

class UkrCVPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="UkrCVPairClassification",
        description="Classifying same or different speaker (12 speakers), Ukrainian",
        dataset={
            "path": "vbrydik/ukr-cv-speakers-v0",
            "revision": "b388f32de54e93d4658c32a7d398a13a00410b89"
        },
        type="AudioPairClassification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["ukr"],
        main_score="max-ap",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        modalities=["audio"],
        sample_creation="created",
        descriptive_stats={
            "n_samples": {"train": 4200}
        },
    )

    # Override default column name in the subclass
  
    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 1000 # guess, fill in later

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset['test'])

        df = df.rename(columns={"speaker_id": "label"})
        grouped = [df.loc[df['label'] == label] for label in df['label'].unique()]

        similar_pairs = []
        dissimilar_pairs = []

        for group in grouped:
            files = [audio['array'].tolist() for audio in group['audio']]
            random.shuffle(files)
            # print(files[0])
            similar_pairs.extend([[files[i], files[i+1], [1]] for i in range(0, len(files) - 1, 2)])

        all_files = [audio['array'].tolist() for audio in df["audio"]]
        all_labels = df["label"].values.tolist()

        num_similar = len(similar_pairs)
        while len(dissimilar_pairs) < num_similar:
            idx1, idx2 = random.sample(range(len(all_files)), 2)
            if all_labels[idx1] != all_labels[idx2]:  
                dissimilar_pairs.append([all_files[idx1], all_files[idx2], [0]])

        pairs = similar_pairs + dissimilar_pairs
        random.shuffle(pairs)

        audio1, audio2, label = zip(*pairs)

        # print(label)

        # convert back to HF dataset
        self.dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_dict({
                'audio1': list(audio1),
                'audio2': list(audio2),
                'label': list(label)
            })
        })

        # res_df = pd.DataFrame(pairs, columns=["audio1", "audio2", "labels"])
        # self.dataset = datasets.DatasetDict({'test': datasets.Dataset.from_pandas(res_df, split='test')})