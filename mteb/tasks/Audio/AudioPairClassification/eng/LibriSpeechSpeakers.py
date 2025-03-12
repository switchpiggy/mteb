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

class LibriSpeechSpeakersPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="LibriSpeechSpeakersPairClassification",
        description="Classifying same or different speaker among 61 speakers",
        reference="https://ieeexplore.ieee.org/document/7178964",
        dataset={
            "path": "sanchit-gandhi/librispeech-data",
            "revision": "48657ef9dfb9b58b680e727617a61323da586985"
        },
        type="AudioPairClassification",
        category="t2t",
        eval_splits=["test.clean"],
        eval_langs=["eng-latn"],
        main_score="max-ap",
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@INPROCEEDINGS{7178964,
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Librispeech: An ASR corpus based on public domain audio books}, 
  year={2015},
  volume={},
  number={},
  pages={5206-5210},
  keywords={Resource description framework;Genomics;Bioinformatics;Blogs;Information services;Electronic publishing;Speech Recognition;Corpus;LibriVox},
  doi={10.1109/ICASSP.2015.7178964}}
        """,
        descriptive_stats={
            "n_samples": {"train": 2620}
        },
    )

    # Override default column name in the subclass
  
    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 1000 # guess, fill in later

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset['test.clean'])

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