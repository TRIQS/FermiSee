
import json
from json import JSONEncoder
import numpy as np

from pythtb import * 

class Encoder(JSONEncoder):
    '''
    The default JSON encoder does not handle numpy objects or complex numbers
    so this overloads the default function to handle storing complex numbers as
    a dictionary and converts every numpyarray instance as a python list when writing the json file.
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"array":True, "list":obj.tolist()}
        if isinstance(obj, complex):
            return {"complex":True, "real":obj.real, "imag":obj.imag}
        return JSONEncoder.default(self, obj)

def decode(d):
    '''
    When decoding a json file this will check for the existence of complex
    number dictionary objects created by the encoder and translates it into a
    complex number.
    '''
    if "complex" in d:
        return complex(d["real"], d["imag"])
    if "array" in d:
        return np.array(d["list"])
    return d


def store(tb_model, file_name):
    '''
    Stores a pythTB object as a .json file with the filename provided.
    An addition could be a path variable to specify the directory where the filed is stored
    
    NOTE: When the object is created pythTB might print some statements about some
    values not being specified, ignore these because those values are added
    afterwards.

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
        json.dump(tb_model.__dict__, write_file, cls=Encoder)
    print('Saved to file: ', file_name)

    return

def load(file_name):
    '''
    Stores a pythTB object as a .json file with the filename provided.
    An addition could be a path variable to specify the directory where the filed is stored

    Parameters
    ----------
    file_name: string
        This is the name of the .json file that will be read 
        Currently the function does not check the type

    Returns
    -------
    model: tb_model object
        pythTB model is constructed and attributes are set to the values loaded
        from the json file

    '''
    f = open(file_name, "r")
    data = json.loads(f.read(), object_hook=decode)
    print("Read in data from:",file_name)

    #initialize the tb model object
    model = tb_model(data["_dim_k"],data["_dim_r"])

    #set all tb model attributes to the corresponding value in data
    for key in model.__dict__:
        model.__dict__[key] = data[key]

    return model
