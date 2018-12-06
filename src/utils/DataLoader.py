from pkg_resources import resource_dir


def get_relative_path(package: str):
    '''
    :param package: the directory to load (e.g. 'data.A1.train')
    :return: the relative path of the directory specified by the package path
    '''
    return resource_dir(str)