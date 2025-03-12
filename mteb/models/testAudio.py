from __future__ import annotations
from mteb import get_model, get_tasks
from mteb.evaluation import MTEB

model_names = ["facebook/wav2vec2-xls-r-300m", "facebook/wav2vec2-xls-r-1b", "facebook/wav2vec2-xls-r-2b",
"microsoft/wavlm-base", "microsoft/wavlm-large", "microsoft/wavlm-base-plus",
"openai/whisper-base", "openai/whisper-large-v3-turbo", "openai/whisper-small"]

for model_name in model_names:
    print('Getting model... ')
    model = get_model(model_name)
    print('done!')
    print('Getting task...')
    tasks = get_tasks(tasks=["VocalSoundPairClassification", "NMSQAPairClassification"])
    print('done!')
    print('Getting evaluator...')
    evaluation = MTEB(tasks=tasks)
    print('done!')
    print('Evaluating: ')
    results = evaluation.run(model, verbosity = 3)
    print('done!')