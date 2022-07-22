using ACE
using ACEatoms
using ACE, ACEatoms, JuLIP, ACEbase
using ACE: save_json, load_json
using JuLIP: AtomicNumber
maxorder = 3
maxdeg = 4
path = "./bases/onsite"
basis = read_dict(load_json(string(path,"/test-max-",maxorder,"maxdeg-",maxdeg,".json")))