import  re, glob

import glob
path = 'ep_raw/*.txt'
files = glob.glob(path)

for file in files:

    raw_data = open(file, "r", errors = "ignore")

    out_file = open(file.replace(".txt", "").replace("ep_raw\\", "ep\\")+"_text.txt", "w")

    for row in raw_data:

        line = re.sub("<[^<]+", "", row)

        if line == '\n':
            continue

        line_len = len(line)
        if line_len > 2:
            if line[line_len - 2] not in ('.', '?', '!', ':'):
                line = line.replace('\n','.\n')

        out_file.write(line)

    raw_data.close()
    out_file.close()