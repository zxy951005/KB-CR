# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util_KB as util
import coref_ops
import conll
import metrics


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)       
        self.KB_embeddings = util.EmbeddingDictionary(config["KB_embeddings"])
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        #self.genres = {g: i for i, g in enumerate(config["genres"])}
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None  # Load eval data lazily.       
        
        if config["bert_path"]:
            self.bert_file = h5py.File(self.config["bert_path"], "r")
        else:
            self.bert_file = None
        self.bert_size = self.config["bert_size"]
         
        self.omcs_k = self.config["omcs_k"]
        self.organisms_k = self.config["organisms_k"]
        self.proteins_k = self.config["proteins_k"]
        self.mygenes_k = self.config["mygenes_k"]
        self.genes_k = self.config["genes_k"]
        self.interaction_types_k = self.config["interaction_types_k"]
             
        self.kko = self.config["coarsekomcs"]

        self.kbmaxnum = self.kko+self.organisms_k+self.proteins_k+self.mygenes_k + self.genes_k+self.interaction_types_k
        self.kbsize=self.kbmaxnum*self.KB_embeddings.size


        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        # Context embeddings.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))
        # Head embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))
        # LM embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))
        input_props.append((tf.float32, [None, None, self.bert_size]))  # bert embeddings.
        # Character indices.
        input_props.append((tf.int32, [None, None, None]))
        input_props.append((tf.int32, [None]))  # Text lengths.
 
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.
        input_props.append((tf.int32, [None]))  # Number IDs.
        input_props.append((tf.int32, [None]))  # MM  ????
        # kb.
        input_props.append((tf.float32, [None, None, self.omcs_k, self.KB_embeddings.size]))
        input_props.append((tf.float32, [None, None, self.organisms_k, self.KB_embeddings.size]))
        input_props.append((tf.float32, [None, None, self.proteins_k, self.KB_embeddings.size]))
        input_props.append((tf.float32, [None, None, self.mygenes_k, self.KB_embeddings.size]))
        input_props.append((tf.float32, [None, None, self.genes_k, self.KB_embeddings.size]))
        input_props.append((tf.float32, [None, None, self.interaction_types_k, self.KB_embeddings.size]))
        #allkb
        input_props.append((tf.float32, [None, None, self.kbsize]))
        
       

        self.queue_input_tensors = [tf.placeholder( dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()
        
        #print("000000000000000000")
        self.predictions, self.loss = self.get_predictions_and_loss( *self.input_tensors)
       
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)        
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline)for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)
        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [
            v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(
            self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0]for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb
        
    def load_bert_embeddings(self, doc_key):
        if self.bert_file is None:
            return np.zeros([0, 0, self.bert_size])
        file_key = doc_key.replace("/", ":")
        group = self.bert_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        bert_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.bert_size])
        for i, s in enumerate(sentences):
            bert_emb[i, :s.shape[0], :] = s
        return bert_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
               
        OMCS = example["OMCS"]
        organisms = example["organisms"]
        proteins = example["proteins"]
        mygenes = example["mygenes"]
        genes = example["genes"]
        interaction_types = example["interaction_types"]
        
        
        
        num_words = sum(len(s) for s in sentences)
        #speakers = util.flatten(example["speakers"])

        #assert num_words == len(speakers)
        
        MMs = {n: i for i, n in enumerate(self.config["MMS"])}
        MM = example["meta_map"]
        MM_ids=[]
        for s in MM: 
            ss=s.split(",")
            if len(ss)==1:
                MM_ids.append(MMs[ss[0]])
                #print("000000")
                #print(MMs[ss[0]])
            else:
                try:
                    mins=min(ss)
                    #print("1111111")
                    #print(mins)
                    MM_ids.append(MMs[mins])
                except:
                    print("2222222")
                    print(s)
        MM_ids=np.array(MM_ids)
        
        
        numbers = {n: i for i, n in enumerate(self.config["numbers"])}
        number = example["number"]
        number_ids = np.array([numbers[s] for s in number])
        
        
        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s)for s in sentences), max(self.config["filter_widths"]))
               
        max_OMCS_length = max(max(len(os) for os in line) for line in OMCS)
        max_organisms_length = max(max(len(og) for og in line) for line in organisms) 
        max_proteins_length = max(max(len(pt) for pt in line) for line in proteins)
        max_mygenes_length = max(max(len(mg) for mg in line) for line in mygenes)
        max_genes_length = max(max(len(gs) for gs in line) for line in genes)
        max_interaction_types_length = max(max(len(tp) for tp in line) for line in interaction_types)
        maxallkb=max_OMCS_length+max_proteins_length+max_mygenes_length+max_genes_length +max_interaction_types_length 
        
               
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        
        
        context_OMCS_emb = np.zeros([len(sentences), max_sentence_length, max_OMCS_length, self.KB_embeddings.size])
        context_organisms_emb = np.zeros([len(sentences), max_sentence_length, max_organisms_length, self.KB_embeddings.size])
        context_proteins_emb = np.zeros([len(sentences), max_sentence_length, max_proteins_length, self.KB_embeddings.size])
        context_mygenes_emb = np.zeros([len(sentences), max_sentence_length, max_mygenes_length, self.KB_embeddings.size])
        context_genes_emb = np.zeros([len(sentences), max_sentence_length, max_genes_length, self.KB_embeddings.size])
        context_interaction_types_emb = np.zeros([len(sentences), max_sentence_length, max_interaction_types_length, self.KB_embeddings.size])
        
        z=[]
        for line in organisms:
            for og in line:
                for gg in og:
                    z.append(len(gg.replace("(","").replace(")","").replace(",","").split(" ")))
        maxog=max(z)
                
        z=[]
        for line in proteins:
            for og in line:
                for gg in og:
                    z.append(len(gg.replace("(","").replace(")","").replace(",","").split(" ")))
        maxpt=max(z) 
        
        z=[]
        for line in mygenes:
            for og in line:
                for gg in og:
                    z.append(len(gg.replace("(","").replace(")","").replace(",","").split(" ")))
        maxmg=max(z)
        
        z=[]
        for line in interaction_types:
            for og in line:
                for gg in og:
                   
                    z.append(len(gg.replace("(","").replace(")","").replace(",","").split(" ")))
        maxtp=max(z)  
        
        tmp_organisms_emb = np.zeros([len(sentences), max_sentence_length, max_organisms_length,maxog, self.KB_embeddings.size])
        tmp_proteins_emb = np.zeros([len(sentences), max_sentence_length, max_proteins_length, maxpt,self.KB_embeddings.size])
        tmp_mygenes_emb = np.zeros([len(sentences), max_sentence_length, max_mygenes_length, maxmg,self.KB_embeddings.size])
        tmp_interaction_types_emb = np.zeros([len(sentences), max_sentence_length, max_interaction_types_length, maxtp,self.KB_embeddings.size])      
                
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c]for c in word]

       
        for i, line in enumerate(OMCS):            
            for j, word in enumerate(line):
                for jj, o in enumerate(word):
                    # snum,maxslen,maxkb,300
                    context_OMCS_emb[i, j, jj] = self.KB_embeddings[o]
                        
              
        for i, line in enumerate(genes):
            for j, word in enumerate(line):
                #print(word)
                for jj, o in enumerate(word):
                    #print(o)
                    # snum,maxslen,maxkb,300
                    context_genes_emb[i, j, jj] = self.KB_embeddings[o.replace("(","").replace(")","").replace(";","").replace(",","")]
                    
        for i, line in enumerate(organisms):
            for j, word in enumerate(line):
                for jj, o in enumerate(word):
                    for jjj, gx in enumerate(o.replace("(","").replace(")","").replace(",","").split(" ")):
                        tmp_organisms_emb[i, j, jj,jjj] = self.KB_embeddings[gx]
        context_organisms_emb=np.sum(tmp_organisms_emb,3)/tmp_organisms_emb.shape[3]
        
        for i, line in enumerate(proteins):
            for j, word in enumerate(line):
                for jj, o in enumerate(word):
                    for jjj, gx in enumerate(o.replace("(","").replace(")","").replace(",","").split(" ")):
                        tmp_proteins_emb[i, j, jj,jjj] =self. KB_embeddings[gx]
        context_proteins_emb=np.sum(tmp_proteins_emb,3)/tmp_proteins_emb.shape[3]
        
        for i, line in enumerate(mygenes):
            for j, word in enumerate(line):
                for jj, o in enumerate(word):
                    for jjj, gx in enumerate(o.replace("(","").replace(")","").replace(";","").replace(",","").split(" ")):
                        tmp_mygenes_emb[i, j, jj,jjj] = self.KB_embeddings[gx]
        context_mygenes_emb=np.sum(tmp_mygenes_emb,3)/tmp_mygenes_emb.shape[3]
        
        for i, line in enumerate(interaction_types):
            for j, word in enumerate(line):
                for jj, o in enumerate(word):
                    for jjj, gx in enumerate(o.split(" ")):
                        tmp_interaction_types_emb[i, j, jj,jjj] = self.KB_embeddings[gx]
        context_interaction_types_emb=np.sum(tmp_interaction_types_emb,3)/tmp_interaction_types_emb.shape[3]
        
        k_OMCS =self.omcs_k 
        k_organisms=self.organisms_k
        k_proteins=self.proteins_k
        k_mygenes=self.mygenes_k
        k_genes=self.genes_k
        k_interaction_types=self.interaction_types_k

        def buzu(max_length,k,emb):
            if max_length<k:
                bo=np.zeros([len(sentences), max_sentence_length, k-max_length, self.KB_embeddings.size])
                emb=np.concatenate((emb,bo),2)
            
            else:
                emb = emb[:,:,:k,:]
            return emb
                
        context_OMCS_emb=buzu(max_OMCS_length,k_OMCS,context_OMCS_emb)        
        context_organisms_emb=buzu(max_organisms_length,k_organisms,context_organisms_emb)
        context_proteins_emb=buzu(max_proteins_length,k_proteins,context_proteins_emb)
        context_mygenes_emb=buzu(max_mygenes_length,k_mygenes,context_mygenes_emb)
        context_genes_emb=buzu(max_genes_length,k_genes,context_genes_emb)
        context_interaction_types_emb=buzu(max_interaction_types_length,k_interaction_types,context_interaction_types_emb)
            
      
        c=[context_OMCS_emb,context_organisms_emb,context_proteins_emb,context_mygenes_emb,context_genes_emb,context_interaction_types_emb]    
        context_allkb_emb =np.concatenate(c,2)
        context_allkb_emb=context_allkb_emb.reshape((len(sentences), max_sentence_length,-1))
        context_allkb_emb=context_allkb_emb[:, :, :self.kbsize]
        
        
        tokens = np.array(tokens)
        #speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        #speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        #genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        lm_emb = self.load_lm_embeddings(doc_key)
        bert_emb = self.load_bert_embeddings(doc_key)
        
        #print("aaaaaaaaaaaaaaaaa")
        #print(len(context_OMCS_emb[0][0]))
        #print(len(context_NELL_emb[0][0]))
        #print(len(context_WORDNET_emb[0][0]))
        #print(len(context_allkb_emb[0][0]))
        #print(doc_key)
        #print(k_OMCS, k_organisms, k_proteins,k_mygenes,k_genes,k_interaction_types)

     
        example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, bert_emb,char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids, number_ids, MM_ids, context_OMCS_emb, context_organisms_emb, context_proteins_emb, context_mygenes_emb, context_genes_emb, context_interaction_types_emb, context_allkb_emb)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors


    def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, bert_emb,char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids, number_ids, MM_ids,OMCS_emb, organisms_emb, proteins_emb, mygenes_emb, genes_emb,interaction_types_emb,allkb_emb):
    
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset +max_training_sentences].sum()
        tokens = tokens[sentence_offset:sentence_offset +max_training_sentences, :]
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        lm_emb = lm_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        bert_emb = bert_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        char_index = char_index[sentence_offset:sentence_offset +max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        # 添加的内容�?        
        OMCS_emb = OMCS_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        organisms_emb = organisms_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        proteins_emb = proteins_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        mygenes_emb = mygenes_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        genes_emb = genes_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        interaction_types_emb = interaction_types_emb[sentence_offset:sentence_offset +max_training_sentences, :, :, :]
        allkb_emb = allkb_emb[sentence_offset:sentence_offset +max_training_sentences, :, :]
        # 添加的内容�?
        #speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        number_ids = number_ids[word_offset: word_offset + num_words]
        MM_ids = MM_ids[word_offset: word_offset + num_words]
        gold_spans = np.logical_and( gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        
        return  tokens, context_word_emb, head_word_emb, lm_emb, bert_emb,char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids, number_ids, MM_ids,OMCS_emb, organisms_emb, proteins_emb, mygenes_emb, genes_emb,interaction_types_emb, allkb_emb

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
      
        same_span = tf.logical_and(same_start, same_end)
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
#         return 1 - (tf.to_float(is_training) * dropout_rate)
        return 1 - (tf.cast(is_training, tf.float32) * dropout_rate)

    # 👇为了kb添加的，返回筛选之后的三个kb的词向量
    def coarse_to_fine_pruning_kb(self, context_word_emb, context_OMCS_emb,kko):

        
        num_sentences = util.shape(context_word_emb,0)
        max_sentence_length = util.shape(context_word_emb,1)
        emb_size = util.shape(context_word_emb,2)
                
        context_word_emb = tf.expand_dims(context_word_emb, -2)

        context_word_OMCS_emb = tf.concat([context_word_emb, context_OMCS_emb], -2)
 
        word_OMCS_scores = tf.keras.layers.Dense(1)(context_word_OMCS_emb)
    

        def filter_kb(kb_emb, kb_scores, k_kb):
            # indices.shape: [num_sentences, max_sentence_length, k_kb]
            values, indices = tf.nn.top_k(tf.squeeze(kb_scores, -1), k_kb)
            # [num_sentences, max_sentence_length, k_kb,1]
            indices_new = tf.expand_dims(indices, -1)
            
            sentence_indices_tmp = tf.tile(tf.range(num_sentences), [max_sentence_length*k_kb])
            sentence_indices = tf.reshape(tf.transpose(tf.reshape(sentence_indices_tmp, [-1, num_sentences])), tf.shape(indices_new))

            word_indices_tmp = tf.tile(tf.range(num_sentences*max_sentence_length) % max_sentence_length, [k_kb])
            word_indices = tf.reshape(tf.transpose(tf.reshape(word_indices_tmp, [-1, num_sentences*max_sentence_length])), tf.shape(indices_new))

            indices_new = tf.concat([sentence_indices, word_indices, indices_new], -1)
            return tf.gather_nd(kb_emb, indices_new)    #[num_sentences, max_sentence_length, k_kb,300]

        OMCS_emb_filtered = filter_kb(context_OMCS_emb, word_OMCS_scores, kko)
       

        return OMCS_emb_filtered
    # 👆为了kb添加�?
    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)

        # c是选择的top_antecedents限制个数 
        c = tf.minimum(self.config["max_top_antecedents"], k)
        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])  # [k, c]
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets  # [k, c]
        top_antecedents_mask = raw_top_antecedents >= 0  # [k, c]
        top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents)  # [k, c]
    
        top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets


    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, bert_emb,char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids, number_ids, MM_ids,OMCS_emb, organisms_emb, proteins_emb, mygenes_emb, genes_emb, interaction_types_emb, allkb_emb):
    
               
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout( self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = util.shape(context_word_emb,0)
        max_sentence_length = util.shape(context_word_emb,1)
        
        if 0 < self.config["top_kb_ratio"] < 1:
            OMCS_emb= self.coarse_to_fine_pruning_kb(context_word_emb, OMCS_emb,self.kko)
        elif self.config["top_kb_ratio"] >= 1 or self.config["top_kb_ratio"] <= 0:
            print("Wrong parameter top_kb_ratio: reset to 1") 
          
        '''
        print("66666666666666")
        print(OMCS_emb.get_shape())
        print(organisms_emb.get_shape())
        print(proteins_emb.get_shape())
        print(mygenes_emb.get_shape())
        print(genes_emb.get_shape())
        print(interaction_types_emb.get_shape())
        '''
        
        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]
                
        def ave(emb):
            emb_sum = tf.reduce_sum(emb, 2)
            indics = tf.cast(tf.reshape(tf.count_nonzero(emb, 2)[:, :, 0], [num_sentences, max_sentence_length, 1]),dtype=tf.float32)  # nums,maxslen,1
            div=tf.where(tf.is_nan(tf.divide(emb_sum, indics)), tf.zeros_like(tf.divide(emb_sum, indics)), tf.divide(emb_sum, indics))
            emb_ave = tf.reshape(div, [num_sentences, max_sentence_length, 1, self.KB_embeddings.size]) 
            
            return  emb_ave
        
        
        if self.config["noKB"]:                   
            KB_emb=None
        
        if self.config["average_KB"]:
            
            OMCS_emb_ave = ave(OMCS_emb) 
            organisms_emb_ave = ave(organisms_emb)
            proteins_emb_ave = ave(proteins_emb)
            mygenes_emb_ave = ave(mygenes_emb)
            genes_emb_ave = ave(genes_emb)
            interaction_types_emb_ave = ave(interaction_types_emb)
            KB_emb_list = [OMCS_emb_ave,organisms_emb_ave ,proteins_emb_ave,mygenes_emb_ave,genes_emb_ave,interaction_types_emb_ave]

            KB_emb = tf.concat(KB_emb_list, 2)  # snum,maxslen,3,300
            KB_emb = tf.to_float(tf.reduce_sum(KB_emb, 2)/3, name='ToFloat')# snum,maxslen,300
            
                   
        if self.config["attention_KB"]:        
            KB_emb_list = [OMCS_emb,organisms_emb, proteins_emb,mygenes_emb,genes_emb,interaction_types_emb]
            all=tf.concat(KB_emb_list, 2)
            allkb_emb=tf.reshape(all,[num_sentences, max_sentence_length,self.kbsize])
            KB_emb =tf.to_float(allkb_emb, name='ToFloat')
            

        if self.config["char_embedding_size"] > 0:
            
            char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index)
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)])  
            
            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"])
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)])  
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        if not self.lm_file:
            elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
            lm_embeddings = elmo_module(
                inputs={"tokens": tokens, "sequence_len": text_len},
                signature="tokens", as_dict=True)
            # [num_sentences, max_sentence_length, 512]
            word_emb = lm_embeddings["word_emb"]
            lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                               lm_embeddings["lstm_outputs1"],
                               lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(tf.get_variable( "lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable( "lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape( lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims( self.lm_weights, 1))  # [num_sentences * max_sentence_length * emb, 1]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)
        
        bert_emb_size = util.shape(bert_emb, 2)
        with tf.variable_scope("bert_aggregation"):
            self.bert_weights = tf.nn.softmax(
                tf.get_variable("bert_scores", [1], initializer=tf.constant_initializer(0.0)))
            self.bert_scaling = tf.get_variable("bert_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_bert_emb = tf.reshape(bert_emb, [num_sentences * max_sentence_length * bert_emb_size, 1])
        flattened_aggregated_bert_emb = tf.matmul(flattened_bert_emb, tf.expand_dims(self.bert_weights,
                                                                                     1))  # [num_sentences * max_sentence_length * emb, 1]
        aggregated_bert_emb = tf.reshape(flattened_aggregated_bert_emb,
                                         [num_sentences, max_sentence_length, bert_emb_size])
        aggregated_bert_emb *= self.bert_scaling
        context_emb_list.append(aggregated_bert_emb)

        context_emb = tf.concat(context_emb_list, 2)
        head_emb = tf.concat(head_emb_list, 2)
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)
        
        # 👇，多了KB_emb
        context_outputs = self.lstm_contextualize( context_emb, text_len, text_len_mask, KB_emb)  # [num_words, emb]

        num_words = util.shape(context_outputs, 0)

        #genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len( self.genres), self.config["feature_size"]]), genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [ 1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence( sentence_indices, text_len_mask)  # [num_words]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts +  tf.expand_dims(tf.range(self.max_span_width),0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather( flattened_sentence_indices, candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(  candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)  # [num_candidates]
        
        feature_index = [number_ids,MM_ids]
        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends,feature_index)
        #candidate_span_emb = self.get_span_emb( flattened_head_emb, context_outputs, candidate_starts, candidate_ends)  # [num_candidates, emb]
        candidate_mention_scores = self.get_mention_scores( candidate_span_emb)  # [k, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   util.shape(context_outputs, 0),
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k]
        #top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]
  
        c = tf.minimum(self.config["max_top_antecedents"], k)

        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
                top_span_emb, top_span_mention_scores, c)

        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets)  # [k, c]
                top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1)  # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1)  # [k, emb]
                with tf.variable_scope("f"):
                    f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1)))  # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))
        same_cluster_indicator = tf.equal( top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]
        loss = tf.reduce_sum(loss)  # []
  
        
        
        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss
    
    def get_feature_embs(self, feature_embeddings, feature_index, span_indices, span_mask):
        feature = tf.gather(feature_embeddings, feature_index)  # num_wordsx20 362x20
        feature = tf.gather(feature, span_indices)  # [k,maxspanwidth,20] 4067x30x20
        feature_emb = feature * span_mask  # 4067x30x20 [k, max_span_width,20]
        feature_sum = tf.reduce_sum(feature_emb, 1)  # 4067x20[k,20]
        feture_nonzero_num = tf.to_float(
            tf.count_nonzero(tf.count_nonzero(feature_emb, 2), 1, keep_dims=True))  # 4067x1[k,1]
        feture_ave = tf.divide(feature_sum, feture_nonzero_num)  # 4067x20[k,20]
        return feture_ave
        
    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends, feature_index):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                   1)  # [k, max_span_width]
        span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
        span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                   2)  # [k, max_span_width, 1]

        if self.config["use_features_span"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)  # [k, emb]
            span_feature_embs = tf.nn.dropout(span_width_emb, self.dropout)

        if self.config["addspan_features"]:
            span_feature_list = []
            feature_number_index = feature_index[0]  # num_words 362
            feature_MM_index = feature_index[1]  # num_words 362

            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),span_width_index)  # [k, emb]
            # span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)

            span_number_embeddings = tf.get_variable("span_number_embeddings",[len(self.config["numbers"]), self.config["feature_size"]])
            number_emb = self.get_feature_embs(span_number_embeddings, feature_number_index, span_indices, span_mask)
            
            span_MM_embeddings = tf.get_variable("span_MM_embeddings",[len(self.config["MMS"]), self.config["feature_size"]])
            MM_emb = self.get_feature_embs(span_MM_embeddings, feature_MM_index, span_indices, span_mask)

            span_feature_list.append(span_width_emb)
            span_feature_list.append(number_emb)
            span_feature_list.append(MM_emb)

            span_feature_emb = tf.concat(span_feature_list, 1)  # 4067x40
            span_feature_embs = tf.nn.dropout(span_feature_emb, self.dropout)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                       1)  # [k, max_span_width]
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            #span_emb_list.append(span_head_emb)
            
        span_emb_list.append(span_feature_embs)
        span_emb_list.append(span_head_emb)
        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]


    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            # [k, 1]
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        '''
        if self.config["use_metadata"]:
          top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
          same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
          speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
          feature_emb_list.append(speaker_pair_emb)

          #tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
          #feature_emb_list.append(tiled_genre_emb)
        '''
        if self.config["use_features_pair"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]
    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    # 👇，多了个参数KB_emb
    def lstm_contextualize(self, text_emb, text_len, text_len_mask, KB_emb):
        num_sentences = tf.shape(text_emb)[0]      
        contxt_embsize = text_emb.get_shape()[2].value

        if self.config["noKB"]:
            current_inputs = text_emb
            maxkb=0
        else:
            maxkb = self.kbmaxnum
            #maxkb = tf.div(util.shape(KB_emb,2),self.KB_embeddings.size)
            current_inputs = tf.concat([text_emb, KB_emb], 2)
            print(text_emb.get_shape())
            print(KB_emb.get_shape())
            print(current_inputs.get_shape())

            
            
        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(
                        self.config["contextualization_size"], num_sentences, self.lstm_dropout, contxt_embsize, maxkb,self.config,layer)  # forward
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(
                        self.config["contextualization_size"], num_sentences, self.lstm_dropout, contxt_embsize, maxkb,self.config,layer)  # backward
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [
                                                         num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))  # forward
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [
                                                         num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))  # backward
                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                              cell_bw=cell_bw,
                                                                              inputs=current_inputs,
                                                                              sequence_length=text_len,
                                                                              initial_state_fw=state_fw,
                                                                              initial_state_bw=state_bw)

                # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs
        
        
        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(
                top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters( top_span_starts, top_span_ends, predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters,mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example
            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            num_words = sum(tensorized_example[2].sum()for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()
        
        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            # 👇，最后多了三个_
            _, _, _, _, _, _, _, _,  gold_starts, gold_ends, _, _, _, _ , _, _, _ ,_, _, _= tensorized_example
            feed_dict = {i: t for i, t in zip(
                self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents( top_antecedents, top_antecedent_scores)
            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
        
        #print(coref_predictions)
        summary_dict = {}
        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}%".format(f * 100))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util.make_summary(summary_dict), average_f1
 
