from __future__ import annotations

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata
import random
import datasets
import pandas as pd
import numpy as np
from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)

random.seed(42)

class CommonVoiceAgePairClassification(MultilingualTask, AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="CommonVoiceAgePairClassification",
        description="A subset of Common Voice classifying whether two audio clips are of the same or different age group (twenties, fourties, etc.) within the same language",
        reference="https://arxiv.org/abs/1912.06670",
        dataset={
            "path": "fixie-ai/common_voice_17_0",
            "revision": "34f78a43893414e7b6e271ba94c1d5e05f18b239"
        },
        type="AudioPairClassification",
        category="t2t", # no audio category yet
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-latn"],
            "ru" : ["rus"],
            "br" : ["bre"],
            "gl": ["glg"]
        },
        main_score="max_ap",
        domains=["Spoken"], 
        task_subtypes=["Emotion classification"], # age group classification
        license="not specified",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ardila-etal-2020-common,
    title = "Common Voice: A Massively-Multilingual Speech Corpus",
    author = "Ardila, Rosana  and
      Branson, Megan  and
      Davis, Kelly  and
      Kohler, Michael  and
      Meyer, Josh  and
      Henretty, Michael  and
      Morais, Reuben  and
      Saunders, Lindsay  and
      Tyers, Francis  and
      Weber, Gregor",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.520/",
    pages = "4218--4222",
    language = "eng",
    ISBN = "979-10-95546-34-4",
    abstract = "The Common Voice corpus is a massively-multilingual collection of transcribed speech intended for speech technology research and development. Common Voice is designed for Automatic Speech Recognition purposes but can be useful in other domains (e.g. language identification). To achieve scale and sustainability, the Common Voice project employs crowdsourcing for both data collection and data validation. The most recent release includes 29 languages, and as of November 2019 there are a total of 38 languages collecting data. Over 50,000 individuals have participated so far, resulting in 2,500 hours of collected audio. To our knowledge this is the largest audio corpus in the public domain for speech recognition, both in terms of number of hours and number of languages. As an example use case for Common Voice, we present speech recognition experiments using Mozilla`s DeepSpeech Speech-to-Text toolkit. By applying transfer learning from a source English model, we find an average Character Error Rate improvement of 5.99 {\ensuremath{\pm}} 5.48 for twelve target languages (German, French, Italian, Turkish, Catalan, Slovenian, Welsh, Irish, Breton, Tatar, Chuvash, and Kabyle). For most of these languages, these are the first ever published results on end-to-end Automatic Speech Recognition."
}
        """,
        descriptive_stats={
            "n_samples": {"test": 2640} # more samples in full dataset, varies per language
        },
    )

    # Override default column name in the subclass
  
    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 100 # varied

    def dataset_transform(self):
        # get English subset
        df = pd.DataFrame(self.dataset['test'])

        df = df[df['age'].notna() & (df['age'] != "")] # remove all data points with empty age labels

        unique_ages = df['age'].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_ages)}

        df['age_id'] = df['age'].map(label_mapping)

        df = df.rename(columns={"age_id": "label"})
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
        print('Number of pairs: ', len(pairs))
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