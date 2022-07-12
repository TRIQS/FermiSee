
import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    '''
    The default JSON encoder does not handle numpy objects so this overloads the default function
    and converts every numpyarray instance as a python list when writing the json file.
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def store(tb_model, file_name):
    '''
    Stores a pythTB object as a .json file with the filename provided.
    An addition could be a path variable to specify the directory where the filed is stored

    Parameters
    ----------
    tb_model: pythTB tight binding model object

    file_name: string
        This is the name of the .json file that will be generated.
        Currently the function does not append ".json" at the end of the filename nor does it check the type

    Returns
    -------

    '''
    if not isinstance(file_name, str):
        raise Exception("file_name variable must be a string")

    with open(file_name, "w") as write_file:
        json.dump(tb_model.__dict__, write_file, cls=NumpyArrayEncoder)
    print('Saved to file: ', file_name)

    return
