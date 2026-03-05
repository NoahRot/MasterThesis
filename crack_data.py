import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tools.plt_spec import *

def read_crack_data(test_path : str):
    # open the file 
    try:
        file = open(test_path + "/crack_data.txt")
    except:
        print("Unable to open the crack_data file:" + test_path + "/crack_data.txt")
        return
        
    # read the header 
    lines =file.readlines()
    header = lines[0].rstrip().split(" ")
        
    # read the content of the file 
    data = {}
    for i in range(0, len(header)):
        d = []
        for j in range(1, len(lines)):
            d.append(float(lines[j].rstrip().split(" ")[i]))
        data[header[i]] = np.array(d)
        
    return data, header

init_plt(latex=False)
create_sns_palette()

path = "../data/fatigue_precracking/Test 3"

data, header = read_crack_data(path)
print(header)
print(data)

a_max = np.max(np.array([data["a1"], data["a2"]]), axis=0)
a_min = np.min(np.array([data["a1"], data["a2"]]), axis=0)

fig = plt.figure()
ax = fig.subplots()
ax.plot(data["Step"], a_max)
ax.plot(data["Step"], a_min)
ax.set_xlabel("Step")
ax.set_ylabel("$a$ [mm]")

bx = ax.twinx()
bx.plot(data["Step"], data["M_P-P"], 'kx--')
bx.set_ylabel("$M_{P-P}$")

ax.legend(["$a_{max}$", "$a_{min}$"])

plt.show()