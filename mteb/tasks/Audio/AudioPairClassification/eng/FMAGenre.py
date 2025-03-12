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

class FMAGenrePairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="FMAGenrePairClassification",
        description="A subset of FMA classifying whether two audio clips are of the same or different genre (7 genres)",
        reference="https://www.researchgate.net/publication/311458869_FMA_A_Dataset_For_Music_Analysis",
        dataset={
            "path": "rpmon/fma-genre-classification",
            "revision": "beba2205a070a912dfc11a5b61c0473a9cb1fc78"
        },
        type="AudioPairClassification",
        category="t2t", # no audio category yet
        eval_splits=["train"],
        eval_langs=["eng-latn"],
        main_score="max_ap",
        domains=["Spoken"], # no task domain existing for music, probably should add
        task_subtypes=["Emotion classification"], # genre classification
        license="not specified",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@article{article,
author = {Defferrard, Michaël and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
year = {2016},
month = {12},
pages = {},
title = {FMA: A Dataset For Music Analysis},
doi = {10.48550/arXiv.1612.01840}
}
        """,
        descriptive_stats={
            "n_samples": {"train": 6400}
        },
    )

    # Override default column name in the subclass
  
    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 1000 # guess, fill in later

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset['train'])

        df = df.rename(columns={"genre": "label"})
        df['label'] = pd.factorize(df['label'])[0]
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