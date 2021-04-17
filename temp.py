import os
count = {'NOT-ENOUGH-INFO':40000,'REFUTES':40000,'SUPPORTS':40000}
with open('data/fever/fever_train_source_file') as f_source:
    with open('data/fever/fever_train_lbl_file') as f_lbl:
        with open('data/fever_balanced/fever_train_source_file','w') as f_sourcew:
            with open('data/fever_balanced/fever_train_lbl_file','w') as f_lblw:
                for line,lbl in zip(f_source,f_lbl):
                    if count[lbl.strip()] ==  0:
                        continue
                    f_sourcew.write(line)
                    f_lblw.write(lbl)
                    count[lbl.strip()] -= 1
