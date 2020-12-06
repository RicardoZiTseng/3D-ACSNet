import json
import os


class Params(object):
    def __init__(self, param):
        if not isinstance(param, dict):
            raise ValueError(
                "Wrong value type, expected `dict`, but got {}".format(type(param)))
        self.param = param

    def __getattr__(self, name):
        return self.param[name]


def save_params(params: Params, path):
    param_dict = params.param
    param_dict = json.dumps(param_dict)
    f = open(path, 'w')
    f.write(param_dict)
    f.close()


def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print(
                    "WARNING: Folder %s already existed and does not need to be created" % directory)
