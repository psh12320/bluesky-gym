"""Persist completed training aircraft after every rollout, including partial runs."""
import csv
import math
import os
from contextlib import ExitStack
from pathlib import Path
from atc.metrics import METRICS


class TrainingRecords:
    def __init__(self,directory,reward_scale):
        if not math.isfinite(reward_scale) or reward_scale<=0:raise ValueError('Invalid reward scale')
        directory=Path(directory)
        paths=[directory/'training-aircraft.csv',directory/'training-returns.csv']
        if any(p.exists() for p in paths):raise ValueError('Training records already exist')
        self.reward_scale=reward_scale;self.count=0;self.stack=ExitStack()
        try:
            self.streams=[self.stack.enter_context(p.open('x',newline='',encoding='utf-8')) for p in paths]
            self.aircraft=csv.DictWriter(self.streams[0],fieldnames=list(METRICS))
            self.returns=csv.DictWriter(self.streams[1],fieldnames=['completion_index','native_return','unscaled_learning_return','learning_return','shaping_return'])
            self.aircraft.writeheader();self.returns.writeheader();self.flush()
        except BaseException:
            self.stack.close();raise

    def flush(self):
        for stream in self.streams:stream.flush();os.fsync(stream.fileno())

    def append(self,aircraft,returns):
        if len(aircraft)!=len(returns) or len(aircraft)<self.count:
            raise ValueError('Completed aircraft and learning returns are not aligned')
        if len(aircraft)==self.count:return
        for index in range(self.count,len(aircraft)):
            metrics=aircraft[index];learning_return=returns[index]
            self.aircraft.writerow(metrics)
            unscaled=learning_return/self.reward_scale
            self.returns.writerow({'completion_index':index,'native_return':metrics['total_reward'],
                'unscaled_learning_return':unscaled,'learning_return':learning_return,
                'shaping_return':unscaled-metrics['total_reward']})
        self.flush();self.count=len(aircraft)

    def __enter__(self):return self
    def __exit__(self,*args):return self.stack.__exit__(*args)
