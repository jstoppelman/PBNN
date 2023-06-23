#!/usr/bin/env python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import pandas as pd

for f in os.listdir('.'):
    if "event" in f:
        summary = EventAccumulator(f).Reload()

tags = summary.Tags()['scalars']
runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})

for tag in tags:
    event_list = summary.Scalars(tag)
    values = list(map(lambda x: x.value, event_list))
    step = list(map(lambda x: x.step, event_list))

    print(len(step))
    r = {"metric": [tag] * len(step), "value": values, "step": step}
    r = pd.DataFrame(r)
    runlog_data = pd.concat([runlog_data, r])

runlog_data.to_csv("output.csv", index=None)

