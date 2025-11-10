from other_imports import *

def info_message(message, *args, end="\n"):
    print(message.format(*args), end=end)
    
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    return path