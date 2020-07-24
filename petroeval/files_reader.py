import lasio

def read_lasio(path:'str')->'object':

    """
    This function reads well logs in las files
    args:
    	  path(str) :paths to lasio file

    returns: las object

    """
    assert isinstance(path, str), 'File must be in a string'
    assert path.endswith('LAS') or path.endswith('las'), 'File must be a valid Lasio files'

    try:

        return lasio.read(path)

    except FileNotFoundError as err :

        raise err


def read_lasios(*args)->'list':

    """
    This function reads well logs in las files
    args:
    path(str) :paths to lasio file

    returns: a list of las objects

    """

    assert [path.lower().endswith('las') for path in args] ,'Files must be a valid lasio files'

    try:
            files=list()

            for _ in args:

                files.append(lasio.read(_))

            return files

    except Exception as err:

        raise err

