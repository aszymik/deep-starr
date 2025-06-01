from Bio import SeqIO

def append_activity_to_fasta(fasta_path, bed_path, output_path):
    """Appends activity from BED to FASTA headers"""
    
    # Mapping from rsID to activity
    activity_dict = {}
    with open(bed_path, 'r') as bed:
        for line in bed:
            chrom, start, end, rsid, activity = line.strip().split('\t')
            key = f"{chrom}:{start}-{end}_{rsid}"
            activity_dict[key] = activity

    # Rewrite FASTA with activity added to the header
    with open(output_path, 'w') as out_fasta:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            header = record.id  # original header: chrom:start-end(+)_{rsid}
            
            # Strip strand and parentheses from header part before underscore
            parts = header.split('_')
            key_part = parts[0].replace('(+)', '')
            rsid = parts[1] if len(parts) > 1 else ''
            key = f"{key_part}_{rsid}"
            
            activity = activity_dict.get(key)
            if activity is not None:
                new_header = f"{key_part}(+)_{{rsid}}_{activity}".format(rsid=rsid)
                record.id = new_header
                record.description = ''
                SeqIO.write(record, out_fasta, 'fasta')
            else:
                print(f"Warning: No activity found for {header}")