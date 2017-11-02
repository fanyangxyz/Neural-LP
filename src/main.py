import os
import argparse
import time
import tensorflow as tf
import numpy as np
from model import Learner
from data import Data, DataPlus
from experiment import Experiment


class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--no_rules', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)    
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    # data property
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    # model architecture
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    # optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    # evaluation
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    
    d = vars(parser.parse_args())
    option = Option(d)
    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    if option.resplit:
      assert not option.no_extra_facts
    if option.accuracy:
      assert option.top_k == 1
    
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    tf.logging.set_verbosity(tf.logging.ERROR)
       
    if not option.query_is_language:
        data = Data(option.datadir, option.seed, option.type_check, option.domain_size, option.no_extra_facts)
    else:
        data = DataPlus(option.datadir, option.seed)
    print("Data prepared.")

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    if not option.query_is_language:
        option.num_query = data.num_query
    else:
        option.num_vocab = data.num_vocab 
        option.num_word = data.num_word # the number of words in each query

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    
    option.save()
    print("Option saved.")

    learner = Learner(option)
    print("Learner built.")

    saver = tf.train.Saver(max_to_keep=option.max_epoch)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        tf.set_random_seed(option.seed)
        sess.run(tf.global_variables_initializer())
        print("Session initialized.")

        if option.from_model_ckpt is not None:
            saver.restore(sess, option.from_model_ckpt)
            print("Checkpoint restored from model %s" % option.from_model_ckpt)

        data.reset(option.batch_size)
        experiment = Experiment(sess, saver, option, learner, data)
        print("Experiment created.")

        if not option.no_train:
            print("Start training...")
            experiment.train()
        
        if not option.no_preds:
            print("Start getting test predictions...")
            experiment.get_predictions()
        
        if not option.no_rules:
            print("Start getting rules...")
            experiment.get_rules()

        if option.get_vocab_embed:
            print("Start getting vocabulary embedding...")
            experiment.get_vocab_embedding()
            
    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()

