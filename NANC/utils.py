def read_metadata(data_f):
    metadata = {}
    cav_num = 0
    last_line = False
    with open(data_f, 'r') as f:
        for line in f:
            if 'First buffer' in line:
                break
            if not line.startswith('#'):
                last_line = True
            else:
                clean_line = line.lstrip('#').strip()
                
                if last_line == False:
                    if clean_line.startswith('##'):
                        cav_num += 1
                        metadata[f'CAV{cav_num}'] = {}
                        cav_number = clean_line.lstrip('##').split(' ')[2]
                        metadata[f'CAV{cav_num}']['cav_number'] = cav_number

                    if ' : ' in clean_line:
                        key, value = clean_line.split(':', 1)
                        val = value.strip()
                        try:
                            if '.' in val:
                                metadata[f'CAV{cav_num}'][key.strip()] = float(val)
                            else:
                                metadata[f'CAV{cav_num}'][key.strip()] = int(val)
                        except ValueError:
                            metadata[f'CAV{cav_num}'][key.strip()] = val
                else:
                    pv_list = clean_line.split()
                    columns = [pv.split(':')[-2] for pv in pv_list]
                    metadata['columns'] = columns
    return metadata