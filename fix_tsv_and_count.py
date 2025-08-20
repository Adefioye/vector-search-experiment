# fix_tsv_and_log_bad_ids.py

input_file = 'generated_queries/climate-fever_generated_queries_random.tsv'
output_file = 'generated_queries/climate-fever_generated_queries_random_fixed.tsv'
bad_lines_file = 'generated_queries/bad_lines.log'

input_count = 0
output_count = 0
bad_lines = []

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        input_count += 1
        parts = line.strip().split(None, 1)  # split into [id, text] on first whitespace
        if len(parts) == 2:
            qid, text = parts
            fout.write(f"{qid}\t{text}\n")
            output_count += 1
        else:
            bad_lines.append(line.strip())

# Write malformed lines to file
if bad_lines:
    with open(bad_lines_file, 'w') as badf:
        for bad in bad_lines:
            badf.write(bad + "\n")

print(f"✅ Finished processing")
print(f"📄 Total input lines:  {input_count}")
print(f"📄 Valid output lines: {output_count}")
print(f"❌ Malformed lines:     {len(bad_lines)}")

if bad_lines:
    print(f"⚠️  Malformed lines saved to: {bad_lines_file}")
