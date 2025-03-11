from __future__ import annotations
from mteb import get_model, get_tasks
from mteb.evaluation import MTEB

model_names = ["facebook/wav2vec2-xls-r-1b", 
"facebook/wav2vec2-xls-r-2b", 
"facebook/wav2vec2-xls-r-300m", 
"facebook/wav2vec2-large-xlsr-53"]

for model_name in model_names:
    print('Getting model... ')
    model = get_model(model_name)
    print('done!')
    print('Getting task...')
    tasks = get_tasks(tasks=["ESC50_PairClassification"])
    print('done!')
    print('Getting evaluator...')
    evaluation = MTEB(tasks=tasks)
    print('done!')
    print('Evaluating: ')
    results = evaluation.run(model, verbosity = 3)
    print('done!')