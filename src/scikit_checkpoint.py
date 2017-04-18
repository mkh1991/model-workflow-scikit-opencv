import json
import warnings
import numpy as np
import os
import pickle
import uuid
import hashlib
from sklearn.externals import joblib

class ScikitCheckpoint():

    def __init__(self, snapshots_path, label=None, type='pickle'):
        self.snapshots_path = snapshots_path
        self.label = label
        self.type = type

    def save_model(self, model, stats):
        snapshot_id = hashlib.sha1(str(uuid.uuid4()).encode("UTF=8")).hexdigest()[:10]
        snapshot_directory = os.path.join(self.snapshots_path, str(snapshot_id))
        model_filename = os.path.join(self.snapshots_path, str(snapshot_id), 'model.dat')
        stats_filename = os.path.join(self.snapshots_path, str(snapshot_id), 'stats.json')
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

        return True

