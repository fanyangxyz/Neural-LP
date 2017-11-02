import numpy as np 
import os
import copy
from math import ceil
from collections import Counter
                           
def resplit(train, facts, no_link_percent):
    num_train = len(train)
    num_facts = len(facts)
    all = train + facts
    
    if no_link_percent == 0.:
        np.random.shuffle(all)
        new_train = all[:num_train]
        new_facts = all[num_train:]
    else:
        link_cntr = Counter()
        for tri in all:
            link_cntr[(tri[1], tri[2])] += 1
        tmp_train = []
        tmp_facts = []
        for tri in all:
            if link_cntr[(tri[1], tri[2])] + link_cntr[(tri[2], tri[1])] > 1:
                if np.random.random() < no_link_percent:
                    tmp_facts.append(tri)
                else:
                    tmp_train.append(tri)
            else:
                tmp_train.append(tri)
        
        if len(tmp_train) > num_train:
            np.random.shuffle(tmp_train)
            new_train = tmp_train[:num_train]
            new_facts = tmp_train[num_train:] + tmp_facts
        else:
            np.random.shuffle(tmp_facts)
            num_to_fill = num_train - len(tmp_train)
            new_train = tmp_train + tmp_facts[:num_to_fill]
            new_facts = tmp_facts[num_to_fill:]
    
    assert(len(new_train) == num_train)
    assert(len(new_facts) == num_facts)

    return new_train, new_facts

class Data(object):
    def __init__(self, folder, seed, type_check, domain_size, no_extra_facts):
        np.random.seed(seed)
        self.seed = seed
        self.type_check = type_check
        self.domain_size = domain_size
        self.use_extra_facts = not no_extra_facts
        self.query_include_reverse = True

        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")
        
        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)
                
        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        
        if os.path.isfile(os.path.join(folder, "facts.txt")):
            self.facts_file = os.path.join(folder, "facts.txt")
            self.share_db = True
        else:
            self.train_facts_file = os.path.join(folder, "train_facts.txt")
            self.test_facts_file = os.path.join(folder, "test_facts.txt")
            self.share_db = False

        self.test, self.num_test = self._parse_triplets(self.test_file)
        self.train, self.num_train = self._parse_triplets(self.train_file)        
        if os.path.isfile(self.valid_file):
            self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        else:
            self.valid, self.train = self._split_valid_from_train()
            self.num_valid = len(self.valid)
            self.num_train = len(self.train)

        if self.share_db: 
            self.facts, self.num_fact = self._parse_triplets(self.facts_file)
            self.matrix_db = self._db_to_matrix_db(self.facts)
            self.matrix_db_train = self.matrix_db
            self.matrix_db_test = self.matrix_db
            self.matrix_db_valid = self.matrix_db
            if self.use_extra_facts:
                extra_mdb = self._db_to_matrix_db(self.train)
                self.augmented_mdb = self._combine_two_mdbs(self.matrix_db, extra_mdb)
                self.augmented_mdb_valid = self.augmented_mdb
                self.augmented_mdb_test = self.augmented_mdb
        else:
            self.train_facts, self.num_train_fact \
                = self._parse_triplets(self.train_facts_file)
            self.test_facts, self.num_test_fact \
                = self._parse_triplets(self.test_facts_file)
            self.matrix_db_train = self._db_to_matrix_db(self.train_facts)
            self.matrix_db_test = self._db_to_matrix_db(self.test_facts)
            self.matrix_db_valid = self._db_to_matrix_db(self.train_facts)
        
        if self.type_check:
            self.domains_file = os.path.join(folder, "stats/domains.txt")
            self.domains = self._parse_domains_file(self.domains_file)
            self.train = sorted(self.train, key=lambda x: x[0])
            self.test = sorted(self.test, key=lambda x: x[0])
            self.valid = sorted(self.valid, key=lambda x: x[0])
            self.num_operator = 2 * self.domain_size
        else:
            self.domains = None
            self.num_operator = 2 * self.num_relation

        # get rules for queries and their inverses appeared in train and test
        self.query_for_rules = list(set(zip(*self.train)[0]) | set(zip(*self.test)[0]) | set(zip(*self._augment_with_reverse(self.train))[0]) | set(zip(*self._augment_with_reverse(self.test))[0]))
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create a parser that maps numbers to queries and operators given queries"""
        assert(self.num_query==2*len(self.relation_to_number)==2*self.num_relation)
        parser = {"query":{}, "operator":{}}
        number_to_relation = {value: key for key, value 
                                         in self.relation_to_number.items()}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in xrange(self.num_relation):
            d = {}
            if self.type_check:
                for i, o in enumerate(self.domains[query]):
                    d[i] = number_to_relation[o]
                    d[i + self.domain_size] = "inv_" + number_to_relation[o]
            else:
                for k, v in number_to_relation.items():
                    d[k] = v
                    d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser
        
    def _parse_domains_file(self, file_name):
        result = {}
        with open(file_name, "r") as f:
            for line in f:
                l = line.strip().split(",")
                l = [self.relation_to_number[i] for i in l]
                relation = l[0]
                this_domain = l[1:1+self.domain_size]
                if len(this_domain) == self.domain_size:
                    pass
                else:
                    # fill in blanks
                    num_remain = self.domain_size - len(this_domain)
                    remains = [i for i in xrange(self.num_relation) 
                                 if i not in this_domain]
                    pads = np.random.choice(remains, num_remain, replace=False)
                    this_domain += list(pads)
                this_domain.sort()
                assert(len(set(this_domain)) == self.domain_size)
                assert(len(this_domain) == self.domain_size)
                result[relation] = this_domain
        for r in xrange(self.num_relation):
            if r not in result.keys():
                result[r] = np.random.choice(range(self.num_relation), 
                                             self.domain_size, 
                                             replace=False)
        return result
    
    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)
        
        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 3)
                output.append((self.relation_to_number[l[1]], 
                               self.entity_to_number[l[0]], 
                               self.entity_to_number[l[2]]))
        return output, len(output)

    def _split_valid_from_train(self):
        valid = []
        new_train = []
        for fact in self.train:
            dice = np.random.uniform()
            if dice < 0.1:
                valid.append(fact)
            else:
                new_train.append(fact)
        np.random.shuffle(new_train)
        return valid, new_train

    def _db_to_matrix_db(self, db):
        matrix_db = {r: ([[0,0]], [0.], [self.num_entity, self.num_entity]) 
                     for r in xrange(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0].append([head, tail])
            matrix_db[rel][1].append(value)
        return matrix_db

    def _combine_two_mdbs(self, mdbA, mdbB):
        """Assume mdbA and mdbB contain distinct elements."""
        new_mdb = {}
        for key, value in mdbA.items():
            new_mdb[key] = value
        for key, value in mdbB.items():
            try:
                value_A = mdbA[key]
                new_mdb[key] = [value_A[0] + value[0], value_A[1] + value[1], value_A[2]]
            except KeyError:
                new_mdb[key] = value
        return new_mdb

    def _count_batch(self, samples, batch_size):
        relations = zip(*samples)[0]
        relations_counts = Counter(relations)
        num_batches = [ceil(1. * x / batch_size) for x in relations_counts.values()]
        return int(sum(num_batches))

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.train_start = 0
        self.valid_start = 0
        self.test_start = 0
        if not self.type_check:
            self.num_batch_train = self.num_train / batch_size + 1
            self.num_batch_valid = self.num_valid / batch_size + 1
            self.num_batch_test = self.num_test / batch_size + 1
        else:
            self.num_batch_train = self._count_batch(self.train, batch_size)
            self.num_batch_valid = self._count_batch(self.valid, batch_size)
            self.num_batch_test = self._count_batch(self.test, batch_size)

    def train_resplit(self, no_link_percent):
      new_train, new_facts = resplit(self.train, self.facts, no_link_percent)
      self.train = new_train 
      self.matrix_db_train = self._db_to_matrix_db(new_facts)
      
    #########################################################################

    def _subset_of_matrix_db(self, matrix_db, domain):
        subset_matrix_db = {}
        for i, r in enumerate(domain):
            subset_matrix_db[i] = matrix_db[r]
        return subset_matrix_db

    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0]+self.num_relation, 
                                    triplet[2], 
                                    triplet[1])]
        return augmented

    def _next_batch(self, start, size, samples):
        assert(start < size)
        end = min(start + self.batch_size, size)
        if self.type_check:
            this_batch_tmp = samples[start:end]
            major_relation = this_batch_tmp[0][0]
            # assume sorted by relations
            batch_size = next((i for i in range(len(this_batch_tmp)) 
                                if this_batch_tmp[i][0] != major_relation), 
                              len(this_batch_tmp))
            end = start + batch_size
            assert(end <= size)
        next_start = end % size
        this_batch = samples[start:end]
        if self.query_include_reverse:
            this_batch = self._augment_with_reverse(this_batch)
        this_batch_id = range(start, end)
        return next_start, this_batch, this_batch_id
        
    def _triplet_to_feed(self, triplets):
        queries, heads, tails = zip(*triplets)
        return queries, heads, tails

    def next_test(self):
        self.test_start, this_batch, _ = self._next_batch(self.test_start, 
                                                       self.num_test, 
                                                       self.test)
        if self.share_db and self.use_extra_facts:
            matrix_db = self.augmented_mdb_test
        else:
            matrix_db = self.matrix_db_test

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db, 
                                                  self.domains[query])
        return self._triplet_to_feed(this_batch), matrix_db

    def next_valid(self):
        self.valid_start, this_batch, _ = self._next_batch(self.valid_start, 
                                                        self.num_valid,
                                                        self.valid)
        if self.share_db and self.use_extra_facts:
            matrix_db = self.augmented_mdb_valid
        else:
            matrix_db = self.matrix_db_valid

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db, 
                                                  self.domains[query])
        return self._triplet_to_feed(this_batch), matrix_db

    def next_train(self):
        self.train_start, this_batch, this_batch_id = self._next_batch(self.train_start,
                                                        self.num_train,
                                                        self.train)
        
        if self.share_db and self.use_extra_facts:
            extra_facts = [fact for i, fact in enumerate(self.train) if i not in this_batch_id]
            extra_mdb = self._db_to_matrix_db(extra_facts)
            augmented_mdb = self._combine_two_mdbs(extra_mdb, self.matrix_db_train)
            matrix_db = augmented_mdb
        else:
            matrix_db = self.matrix_db_train

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db, self.domains[query])
        
        return self._triplet_to_feed(this_batch), matrix_db


class DataPlus(Data):
    def __init__(self, folder, seed):
        np.random.seed(seed)
        self.seed = seed
        self.kb_relation_file = os.path.join(folder, "kb_relations.txt")
        self.kb_entity_file = os.path.join(folder, "kb_entities.txt")
        self.query_vocab_file = os.path.join(folder, "query_vocabs.txt")

        self.kb_relation_to_number = self._numerical_encode(self.kb_relation_file)
        self.kb_entity_to_number = self._numerical_encode(self.kb_entity_file)
        self.query_vocab_to_number = self._numerical_encode(self.query_vocab_file)

        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        self.facts_file = os.path.join(folder, "facts.txt")

        self.test, self.num_test = self._parse_examples(self.test_file)
        self.train, self.num_train = self._parse_examples(self.train_file)
        self.valid, self.num_valid = self._parse_examples(self.valid_file)
        self.facts, self.num_fact = self._parse_facts(self.facts_file)
        self.all_exams = set([tuple(q + [h, t]) for (q, h, t) in self.train + self.test + self.valid])

        self.num_word = len(self.test[0][0])
        self.num_vocab = len(self.query_vocab_to_number)
        self.num_relation = len(self.kb_relation_to_number)
        self.num_operator = 2 * self.num_relation
        self.num_entity = len(self.kb_entity_to_number)

        self.matrix_db = self._db_to_matrix_db(self.facts)
        self.matrix_db_train = self.matrix_db
        self.matrix_db_test = self.matrix_db
        self.matrix_db_valid = self.matrix_db

        self.type_check = False
        self.domain_size = None
        self.use_extra_facts = False
        self.query_include_reverse = False
        self.share_db = False

        self.parser = self._create_parser()
        #self.query_for_rules = [list(q) for q in Counter([tuple(q) for (q, _, _) in self.test]).keys()]
        self.query_for_rules = [list(q) for q in set([tuple(q) for (q, _, _) in self.test + self.train])]

    def _numerical_encode(self, file_name):
        lines = [l.strip() for l in open(file_name, "r").readlines()]
        line_to_number = {line: i for i, line in enumerate(lines)}
        return line_to_number

    def _parse_examples(self, file_name):
        lines = [l.strip().split("\t") for l in open(file_name, "r").readlines()]
        triplets = [[[self.query_vocab_to_number[w] for w in l[1].split(",")],
                      self.kb_entity_to_number[l[0]],
                      self.kb_entity_to_number[l[2]],]
                    for l in lines]
        return triplets, len(triplets)    

    def _parse_facts(self, file_name):
        lines = [l.strip().split("\t") for  l in open(file_name, "r").readlines()]
        facts = [[self.kb_relation_to_number[l[1]], 
                  self.kb_entity_to_number[l[0]],
                  self.kb_entity_to_number[l[2]]]
                 for l in lines]
        return facts, len(facts)

    def _create_parser(self):
        parser = {"operator":{}}
        number_to_relation = {value: key for key, value 
                                         in self.kb_relation_to_number.items()}
        number_to_query_vocab = {value: key for key, value 
                                            in self.query_vocab_to_number.items()}
    
        parser["query"] = lambda ws: ",".join([number_to_query_vocab[w] for w in ws]) + " "
            
        d = {}
        for k, v in number_to_relation.items():
            d[k] = v
            d[k + self.num_relation] = "inv_" + v
        parser["operator"] = d
        
        return parser

    def is_true(self, q, h, t):
        if tuple(q + [h, t]) in self.all_exams:
            return True
        else:
            return False
          
