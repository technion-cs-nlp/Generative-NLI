data_dir_prefix = './data/snli_1.0/cl_snli'

with open(data_dir_prefix + '_train_source_file') as train_lines_file:
    train_lines = train_lines_file.readlines()

# train_lines_splited = train_lines.split('|||')

with open(data_dir_prefix + '_premise', 'w') as f:
    f.writelines([line.split('|||')[0] for line in train_lines])

with open(data_dir_prefix + '_hypothesis', 'w') as f:
    f.writelines([line.split('|||')[1] for line in train_lines])
