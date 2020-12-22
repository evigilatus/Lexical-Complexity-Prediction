import  re, glob

import glob
path = 'craft_raw/*.txt'
files = glob.glob(path)

for file in files:

    raw_data = open(file, "r", errors = "ignore")

    out_file = open(file.replace(".txt", "").replace("craft_raw\\", "craft\\")+"_t.txt", "w")

    for line in raw_data:

        if line == '\n':
            continue

        line_len = len(line)
        if line_len > 2:
            if line[line_len - 2] not in ('.', '?', '!', ':'):
                continue #title

        out_file.write(line)

    raw_data.close()
    out_file.close()