def run_length_encod(seq):
  compressed = []
  count = 1
  char = seq[0]
  print(char)
  for i in range(1,len(seq)):
    if seq[i] == char:
      count = count + 1
    else :
      compressed.append((char,count))
      char = seq[i]
      count = 1
  compressed.append((char,count))
  return compressed
 
def run_length_decoding(compressed_seq):
  seq = []
  for i in range(0,len(compressed_seq)):
    for j in range(compressed_seq[i][1]):
        seq.append(int(compressed_seq[i][0]))
 
  return(seq)