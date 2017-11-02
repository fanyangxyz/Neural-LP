import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
from utils import list_rules, print_rules


class Experiment():
    """
    This class handles all experiments related activties, 
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules. 

    Args:
        sess: a TensorFlow session 
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can  
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain 
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """
    
    def __init__(self, sess, saver, option, learner, data):
        self.sess = sess
        self.saver = saver
        self.option = option
        self.learner = learner
        self.data = data
        # helpers
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")


    def one_epoch(self, mode, num_batch, next_fn):
        epoch_loss = []
        epoch_in_top = []
        for batch in xrange(num_batch):
            if (batch+1) % max(1, (num_batch / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, num_batch))
                sys.stdout.flush()
            
            (qq, hh, tt), mdb = next_fn()
            if mode == "train":
                run_fn = self.learner.update
            else:
                run_fn = self.learner.predict
            loss, in_top = run_fn(self.sess,
                                  qq, 
                                  hh, 
                                  tt, 
                                  mdb) 
            epoch_loss += list(loss)
            epoch_in_top += list(in_top)
                                    
        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f." 
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        if self.epoch > 0 and self.option.resplit:
            self.data.train_resplit(self.option.no_link_percent)
        loss, in_top = self.one_epoch("train", 
                                      self.data.num_batch_train, 
                                      self.data.next_train)
        
        self.train_stats.append([loss, in_top])
        
    def one_epoch_valid(self):
        loss, in_top = self.one_epoch("valid", 
                                      self.data.num_batch_valid, 
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        loss, in_top = self.one_epoch("test", 
                                      self.data.num_batch_test,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])
    
    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            self.epoch += 1
            model_path = self.saver.save(self.sess, 
                                         self.option.model_path,
                                         global_step=self.epoch)
            print("Model saved at %s" % model_path)
            
            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))
        
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)       
        print(msg)
        self.log_file.write(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))

    def get_predictions(self):
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        for batch in xrange(self.data.num_batch_test):
            if (batch+1) % max(1, (self.data.num_batch_test / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb = self.data.next_test()
            in_top, predictions_this_batch \
                    = self.learner.get_predictions_given_queries(self.sess, qq, hh, tt, mdb)
            all_in_top += list(in_top)

            for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    eval_fn = lambda (j, p): p >= p_head and (j != h)
                elif self.option.rand_break:
                    eval_fn = lambda (j, p): (p > p_head) or ((p == p_head) and (j != h) and (np.random.uniform() < 0.5))
                else:
                    eval_fn = lambda (j, p): (p > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)
                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()
        
        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")

        msg = "Test in top %0.4f" % np.mean(all_in_top)
        msg += self.msg_with_time("\nTest predictions written.")
        print(msg)
        self.log_file.write(msg + "\n")
        
    def get_attentions(self):
        if self.option.query_is_language:
            num_batch = int(np.ceil(1.0*len(self.data.query_for_rules)/self.option.batch_size))
            query_batches = np.array_split(self.data.query_for_rules, num_batch)
        else:   
            #print(self.data.query_for_rules)
            if not self.option.type_check:
                num_batch = int(np.ceil(1.*len(self.data.query_for_rules)/self.option.batch_size))
                query_batches = np.array_split(self.data.query_for_rules, num_batch)       
            else:
                query_batches = [[i] for i in self.data.query_for_rules]

        all_attention_operators = {}
        all_attention_memories = {}

        for queries in query_batches:
            attention_operators, attention_memories \
            = self.learner.get_attentions_given_queries(self.sess, queries)
            
            # Tuple-ize in order to be used as dict keys
            if self.option.query_is_language:
                queries = [tuple(q) for q in queries]

            for i in xrange(len(queries)):
                all_attention_operators[queries[i]] \
                                        = [[attn[i] 
                                        for attn in attn_step] 
                                        for attn_step in attention_operators]
                all_attention_memories[queries[i]] = \
                                        [attn_step[i, :] 
                                        for attn_step in attention_memories]
        pickle.dump([all_attention_operators, all_attention_memories], 
                    open(os.path.join(self.option.this_expsdir, "attentions.pckl"), "w"))
               
        msg = self.msg_with_time("Attentions collected.")
        print(msg)
        self.log_file.write(msg + "\n")

        all_queries = reduce(lambda x,y: list(x) + list(y), query_batches, [])
        return all_attention_operators, all_attention_memories, all_queries

    def get_rules(self):
        all_attention_operators, all_attention_memories, queries = self.get_attentions()

        all_listed_rules = {}
        all_printed_rules = []
        for i, q in enumerate(queries):
            if not self.option.query_is_language:
                if (i+1) % max(1, (len(queries) / 5)) == 0:
                    sys.stdout.write("%d/%d\t" % (i, len(queries)))
                    sys.stdout.flush()
            else: 
                # Tuple-ize in order to be used as dict keys
                q = tuple(q)
            all_listed_rules[q] = list_rules(all_attention_operators[q], 
                                             all_attention_memories[q],
                                             self.option.rule_thr,)
            all_printed_rules += print_rules(q, 
                                             all_listed_rules[q], 
                                             self.data.parser,
                                             self.option.query_is_language)

        pickle.dump(all_listed_rules, 
                    open(os.path.join(self.option.this_expsdir, "rules.pckl"), "w"))
        with open(os.path.join(self.option.this_expsdir, "rules.txt"), "w") as f:
            for line in all_printed_rules:
                f.write(line + "\n")
        msg = self.msg_with_time("\nRules listed and printed.")
        print(msg)
        self.log_file.write(msg + "\n")

    def get_vocab_embedding(self):
        vocab_embedding = self.learner.get_vocab_embedding(self.sess)
        msg = self.msg_with_time("Vocabulary embedding retrieved.")
        print(msg)
        self.log_file.write(msg + "\n")
        
        vocab_embed_file = os.path.join(self.option.this_expsdir, "vocab_embed.pckl")
        pickle.dump({"embedding": vocab_embedding, "labels": self.data.query_vocab_to_number}, open(vocab_embed_file, "w"))
        msg = self.msg_with_time("Vocabulary embedding stored.")
        print(msg)
        self.log_file.write(msg + "\n")

    def close_log_file(self):
        self.log_file.close()
