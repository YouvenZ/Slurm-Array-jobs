
import pickle as pickle

def save_dict(dict_:dict,name:str)->None:
    """
    Do : Save and name a dictionnary in a .pkl extension file  
    Parameters in : dict_, any dict
                    name, string name of the .pkl file 
    Return : None
    """
    with open(name, 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_dict(name:str)->None:
    """
    Do : Load a pickle by giving the name of the pickle file
    Parameters in : name, string of the pickle name file 
    Return : None
    """

    with open(name, 'rb') as handle:
        dict_ = pickle.load(handle)
    return dict_

