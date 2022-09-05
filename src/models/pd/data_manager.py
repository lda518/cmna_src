import json
import os

class Data_manager:
    def __init__(self):
        self.data_direct = os.path.join('.','saved_data')
        if not os.path.exists(self.data_direct):
            os.mkdir(self.data_direct)

    def save(self, filename, data):
        filename = os.path.join(self.data_direct,filename)
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load(self, filename):
        filename = os.path.join(self.data_direct,filename)
        with open(filename, 'r') as read_file:
            data = json.load(read_file)
        return data
