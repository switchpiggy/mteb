
from __future__ import annotations
from mteb import get_model, get_tasks
from mteb.evaluation import MTEB

model_name = "facebook/wav2vec2-xls-r-300m"

model = get_model(model_name)
tasks = get_tasks(tasks=["ESC50_PairClassification"])
evaluation = MTEB(tasks=tasks)
results = evaluation.run(model)

