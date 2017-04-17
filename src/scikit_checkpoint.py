import json
import warnings
import numpy as np
import os
import pickle
from sklearn.externals import joblib

class ScikitCheckpoint():

    def __init__(self, snapshots_path, label=None, type='pickle'):
        self.snapshots_path = snapshots_path
        self.label = label
        self.type = type
        self.count = 0

    def save_model(self, model, stats):
        snapshot_directory = os.path.join(self.snapshots_path, str(self.count))
        model_filename = os.path.join(self.snapshots_path, str(self.count), 'model.dat')
        stats_filename = os.path.join(self.snapshots_path, str(self.count), 'stats.json')
        if not os.path.exists(snapshot_directory):
            os.makedirs(snapshot_directory)
        if self.label:
            stats['label'] = self.label
        if self.type:
            stats['type'] = self.type

        if self.type == 'pickle':
            pickle.dump(model, open(model_filename, 'wb'))
        elif self.type == 'joblib':
            joblib.dump(model, model_filename)

        with open(stats_filename, 'wb') as f:
            f.write(json.dumps(stats))

        self.count += 1

        return True

