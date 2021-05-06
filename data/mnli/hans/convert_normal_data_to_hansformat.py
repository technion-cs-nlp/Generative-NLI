import glob
import os
import shutil

def ph2inpoutp(p, h, sep='--+'):
    shared_tokens = [p_i for p_i in p if p_i in h]
    input_tokens = []
    output_tokens = []
    for p_i in p:
        inp_tok = p_i if p_i in shared_tokens else '<mask>'
        input_tokens.append(inp_tok)
        output_tokens.append(p_i)

    input_tokens.append(sep)
    output_tokens.append(sep)

    for h_i in h:
        inp_tok = h_i if h_i in shared_tokens else '<mask>'
        input_tokens.append(inp_tok)
        output_tokens.append(h_i)

    return input_tokens, output_tokens

filenames = [fn for fn in sorted(glob.glob('../*source_file')) if 'hard' not in fn]

out_dir = 'train_test_data_hansformat'
os.makedirs(out_dir, exist_ok=True)

ph_sep = '|||'
for filename in filenames:
    print(filename)

    output_lines = []
    with open(filename, 'r') as f:
        for line in f:
            p, h = line.strip().split(ph_sep)
            p = p.split(' ')
            h = h.split(' ')

            intok, outtok = ph2inpoutp(p, h)
            intok = ' '.join(intok)
            outtok = ' '.join(outtok)
            output_lines.append(f'{outtok}{ph_sep}{intok}')

    new_fn = f'{out_dir}/{os.path.basename(filename)}'
    with open(new_fn, 'w') as f:
        for line in output_lines:
            f.write(line+'\n')
    
    filename_lbl = filename.replace('source_file', 'lbl_file')
    new_fn_lbl = new_fn.replace('source_file', 'lbl_file')
    shutil.copyfile(filename_lbl, new_fn_lbl)
