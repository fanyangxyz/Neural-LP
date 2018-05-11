import sys
import os
import pickle
from collections import defaultdict

folder_name = sys.argv[1]
all_file = os.path.join(folder_name, "all.txt")
    
facts = []
with open(all_file, "r") as f:
  for line in f:
    l = line.strip().split("\t")
    assert(len(l) == 3)
    facts.append(l)
num_fact = len(facts)
print("Number of all facts %d" % num_fact)

query_head = defaultdict(list)
query_tail = defaultdict(list)
for h, r, t in facts:
  query_head[(r, h)].append(t)
  query_tail[(r, t)].append(h)

to_dump = {}
to_dump["query_head"] = query_head
to_dump["query_tail"] = query_tail
truths_file = os.path.join(folder_name, "truths.pckl")
pickle.dump(to_dump, open(truths_file, "w"))

print("Gather truths done.")
