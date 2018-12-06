from pkg_resources import resource_filename

def get_relative_path(package: str, directory: str):
    '''
    :param package: the directory to load (e.g. 'data.A1', 'train')
    :return: the relative path of the directory specified by the package path
    '''
    return resource_filename(package, directory)


print(get_relative_path(__name__, 'data'))