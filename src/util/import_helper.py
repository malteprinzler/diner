import importlib


def import_from(module, obj_name):
    """
    mimics behavior of 'from <module> import <obj>'
    :param module: 
    :param obj: 
    :return: 
    """
    pkg = importlib.import_module(module)
    obj = pkg.__dict__[obj_name]
    return obj


def import_obj(s: str):
    """
    directly returns object from module. E.g 'path.to.package.object' returns 'object'
    :param s:
    :return:
    """
    module, obj_name = ".".join(s.split(".")[:-1]), s.split(".")[-1]
    obj = import_from(module, obj_name)
    return obj
