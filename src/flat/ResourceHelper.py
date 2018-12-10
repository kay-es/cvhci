from pkg_resources import resource_filename

def get_path(task: str, dataset: str, special: str = ""):
    '''
    :param package: the directory to load (e.g. 'A1', 'train' or 'A2', 'test', 'n')
    :return: the relative path of the directory specified by the package path
    '''
    data = dataset + '/' + special if special != "" else dataset
    return '../resources/' + task + '/data/' + data