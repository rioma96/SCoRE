##########################################################################
########### PREPROCESSING UTILITIES
##########################################################################

import numpy as np
import pandas as pd
import tensorflow as tf

def distant_selector(df,p_distant):
    indices_distant=np.random.choice(df.index,int(len(df)*p_distant),replace=False).tolist()
    distant_mask=df.index.isin(indices_distant)
    df.loc[distant_mask,'link_name']=df.loc[distant_mask,'distant_y']
    return df

def concatenate_arrays(row, cls):
    #return np.concatenate([row['emb_head'], row['emb_tail']],1).flatten()
    if cls:
        return np.concatenate([row['emb_head'], row['emb_tail'], row['emb_cls']])
    else:
        return np.concatenate([row['emb_head'], row['emb_tail']])

def set_minimal_lists(el):
    return el if isinstance(el, list) else [el]

def balance_triples_training_df(df,quantile_cutoff):
    df['link_name']=df['link_name'].sort_values()
    df['link_name_id']=df['link_name'].transform(lambda x: ''.join(x))
    count_htr=df[['syn_id_head','syn_id_tail','link_name_id']].value_counts().reset_index(name='count')
    
    triple_cutoff=int(count_htr['count'].quantile(quantile_cutoff))
    print(f'cutting triples occurring more than {triple_cutoff} times')
    
    over_represented_triples=count_htr.loc[count_htr['count']>triple_cutoff]

    to_drop=[]
    for head,tail,rel,count in over_represented_triples.values.tolist():
        same_triple_indices=df.loc[(df['syn_id_head']==head)&(df['syn_id_tail']==tail)&(df['link_name_id']==rel)].index.tolist()
        to_keep=np.random.choice(same_triple_indices,triple_cutoff,replace=False).tolist()
        to_drop+=list(set(same_triple_indices)-set(to_keep))
        
    df0=df.reindex(list(set(df.index.to_list())-set(to_drop)))

    return df0


def df_raw_rebalancing(df,balance=False):
    df=df.reset_index(drop=True)
    indices = df.index.values
    numpy_mls = np.array(df['link_name'].values.tolist())
    number_els_per_class=numpy_mls.sum(0)
    print(f'original balancing {number_els_per_class}\n\n')
    
    if balance:
        ideal_number_els_per_class=int(number_els_per_class.mean()) #np.max(number_els_per_class)/2
        under_represented_classes=np.where(number_els_per_class<ideal_number_els_per_class)[0]


        to_keep_indices=[]
        count_new_classes=np.zeros(numpy_mls.shape[1])
        for i in range(numpy_mls.shape[1]):
            cl=np.argsort(number_els_per_class)[i]
            cl_indices=indices[numpy_mls[:,cl]>0]

            #print(len(cl_indices))
            cl_indices = set(cl_indices)-set(to_keep_indices)
            cl_indices = np.array(list(set(cl_indices)))
            #print(len(cl_indices))

            cl_indices=np.random.choice(cl_indices,ideal_number_els_per_class)
            to_keep_indices=np.concatenate([to_keep_indices,cl_indices]).astype(int)
        count_new_classes=numpy_mls[to_keep_indices].sum(0)
        print(f'Miracle in progress... \ntransforming an imbalanced dataset of {len(df)} elements \
        \nin a quasi-balanced dataset of {len(to_keep_indices)} elements \n(unique elements {len(set(to_keep_indices))}).\n\n')
        print(f'new balancing {count_new_classes}\n\n')
        min_batch_size=np.ceil(2/min(count_new_classes/count_new_classes.sum()))
        ideal_batch_size=np.ceil(np.min(number_els_per_class)/min(count_new_classes/count_new_classes.sum())) 
        print(f'According to the results a minimum/ideal batch of size {min_batch_size}/{ideal_batch_size} is needed in training\n\n')

        df=df.reindex(to_keep_indices).sample(frac=1).reset_index(drop=True)
        return df, count_new_classes/len(df)
    else:
        return number_els_per_class/len(df)
    

#Split data of embeddings in CLS, Head and Tail
def split_embeddings(df):
    
    # Splitting the list into separate columns
    df[['emb_cls', 'emb_head', 'emb_tail']] = df['embeddings'].apply(pd.Series)
    
    # Dropping the original column with lists
    df.drop('embeddings', axis=1, inplace=True)
    
    df['emb_head'] = df['emb_head'].apply(lambda x: tf.reduce_mean(x, axis=0))
    df['emb_tail'] = df['emb_tail'].apply(lambda x: tf.reduce_mean(x, axis=0))
    df['emb_cls'] = df['emb_cls'].apply(lambda x: x.numpy())
    df['emb_head'] = df['emb_head'].apply(lambda x: x.numpy())
    df['emb_tail'] = df['emb_tail'].apply(lambda x: x.numpy())

    return df 


def from_file_list_to_tfdataset(file_list, 
                                relation_dict=None,
                                balance_triples=False, 
                                quantile_cutoff=0.95,
                                balance_labels=False,
                                get_thresholds=False,
                                shades=False, 
                                p_distant=0.0, 
                                del_empty=False,
                                return_df=False,
                                return_headtail=False,
                                cls=False):
    
    from sklearn.preprocessing import MultiLabelBinarizer
    import itertools

    df = pd.concat([pd.read_pickle(file) for file in file_list], ignore_index=True)
    df = split_embeddings(df)

    # df = df.groupby(['sentence', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end','syn_id_head', 'syn_id_tail']).agg(list)
    df = df.groupby(['syn_id_head', 'syn_id_tail','sentence']).agg(list)

    new_link_name_column = [list(set(itertools.chain(*list_of_iterables))) for list_of_iterables in df.loc[:,"link_name"]]
    df = df.map(lambda x: x[0])
    df["link_name"] = new_link_name_column
    df.reset_index(inplace=True,drop=False)
    
    if del_empty:
        df = df[df['distant_y'].apply(len) > 0]

    if shades:
        df=distant_selector(df,p_distant)
    
    if balance_triples:
        df=balance_triples_training_df(df,quantile_cutoff)
    
    df['Concatenated'] = df.apply(concatenate_arrays, args=(cls,), axis=1)
    df = df[['Concatenated', 'link_name', 'syn_id_head', 'syn_id_tail']]
    
    print("Dataset loaded")
    
    if relation_dict is None:
        codes, relation_dict = pd.factorize(df['link_name'].explode())
        relation_dict = dict([(el, i) for i, el in enumerate(relation_dict)])
        df['link_name']=df['link_name'].map(lambda x: [relation_dict[i] for i in x])
    else:
        # Check if all values in df['link_name'] are in relation_dict
        if all(value in relation_dict for value in df['link_name'].explode().unique()):
            # If all values exist in relation_dict, proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])
        else:
            # Remove rows where at least one value is not in relation_dict
            df = df[df['link_name'].apply(lambda x: all(i in relation_dict for i in x))]
            # Proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])
    list_of_classes=np.array(list(relation_dict.values()))

    one_hot = MultiLabelBinarizer(classes=list_of_classes)

    print("Relations mapped")
    
    df['link_name'] = df['link_name'].apply(set_minimal_lists)
    df['link_name'] = one_hot.fit_transform(df['link_name'].to_list()).tolist()
    
    if balance_labels:
        print('Balancing labels\n\n')
        df , thresholds = df_raw_rebalancing( df , balance=balance_labels )
    else:
        thresholds = df_raw_rebalancing(df)
    
    print("one_hot.fit_transform done")
    print("Starting tf.data.Dataset.from_tensor_slices")
    if return_headtail:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['Concatenated'].tolist()),
            np.array(df['link_name'].tolist()),
            np.array(df[['syn_id_head', 'syn_id_tail']].to_numpy())
        ))
    else:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['Concatenated'].tolist()),
            np.array(df['link_name'].tolist())
        ))

    print("tfdataset created")
    
    to_return = [relation_dict, training_set]
    if return_df:
        to_return = to_return+[df]
    if get_thresholds:
        to_return = [thresholds] + to_return
    return to_return



def filter_single_relation(df,
                           relation_nr=0,
                           return_headtail=False,
                           return_df=False,
                           balance=False
                           ):
    
    df['link_name']=df['link_name'].apply(lambda x: [x[relation_nr]])
   
    if balance:
        print('Balancing labels\n\n')
        df=df.reset_index(drop=True)
        indices = df.index.values
        rel_vec=np.array(df['link_name'].tolist())
        n_rebalance=rel_vec.shape[0]-rel_vec.sum()


        add_indices=np.random.choice(indices[rel_vec[:,0]>0],n_rebalance)
        new_index=np.concatenate([indices,add_indices])
        df=df.reindex(new_index.tolist())
        df=df.sample(frac=1)
    
    
    print(f"Selecting relation {relation_nr}")
    print("Starting tf.data.Dataset.from_tensor_slices")
    if return_headtail:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['Concatenated'].tolist()),
            np.array(df['link_name'].tolist()),
            np.array(df[['syn_id_head', 'syn_id_tail']].to_numpy())
        ))
    else:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['Concatenated'].tolist()),
            np.array(df['link_name'].tolist())
        ))

    print("tfdataset created")
    
   
    if return_df:
        return training_set,df
    else:
        return training_set



def from_file_list_to_df_GS(file_list, 
                                relation_dict=None,
                                get_thresholds=False,
                                del_empty=False,
                                cls=False):
    
    from sklearn.preprocessing import MultiLabelBinarizer
    import itertools

    df = pd.concat([pd.read_pickle(file) for file in file_list], ignore_index=True)
    df = split_embeddings(df)

    # df = df.groupby(['sentence', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end','syn_id_head', 'syn_id_tail']).agg(list)
    df = df.groupby(['syn_id_head', 'syn_id_tail','sentence']).agg(list)

    new_link_name_column = [list(set(itertools.chain(*list_of_iterables))) for list_of_iterables in df.loc[:,"link_name"]]
    df = df.map(lambda x: x[0])
    df["link_name"] = new_link_name_column
    df.reset_index(inplace=True,drop=False)
    
    if del_empty:
        df = df[df['distant_y'].apply(len) > 0]

    
    df['Concatenated'] = df.apply(concatenate_arrays, args=(cls,), axis=1)
    df = df[['Concatenated', 'link_name', 'syn_id_head', 'syn_id_tail']]
    
    print("Dataset loaded")
    
    if relation_dict is None:
        codes, relation_dict = pd.factorize(df['link_name'].explode())
        relation_dict = dict([(el, i) for i, el in enumerate(relation_dict)])
        df['link_name']=df['link_name'].map(lambda x: [relation_dict[i] for i in x])
    else:
        # Check if all values in df['link_name'] are in relation_dict
        if all(value in relation_dict for value in df['link_name'].explode().unique()):
            # If all values exist in relation_dict, proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])
        else:
            # Remove rows where at least one value is not in relation_dict
            df = df[df['link_name'].apply(lambda x: all(i in relation_dict for i in x))]
            # Proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])
    list_of_classes=np.array(list(relation_dict.values()))

    one_hot = MultiLabelBinarizer(classes=list_of_classes)

    print("Relations mapped")
    
    df['link_name'] = df['link_name'].apply(set_minimal_lists)
    df['link_name'] = one_hot.fit_transform(df['link_name'].to_list()).tolist()
    
    thresholds = df_raw_rebalancing(df)
    
    df = df[['Concatenated', 'link_name']]

    print("tfdataset created")
    
    if get_thresholds:
        return thresholds, relation_dict, df
    else:
        return relation_dict, df 


def df_to_tf_dataset(df):
    print("Starting tf.data.Dataset.from_tensor_slices")
    training_set = tf.data.Dataset.from_tensor_slices((
        np.array(df['Concatenated'].tolist()),
        np.array(df['link_name'].tolist())
    ))
    print("tfdataset created")
    return training_set
    
    
def HT_sentence_matching(file_list, 
                         relation_dict=None):
    
    from sklearn.preprocessing import MultiLabelBinarizer
    import itertools

    df = pd.concat([pd.read_pickle(file) for file in file_list], ignore_index=True)
    df = split_embeddings(df)

    # df = df.groupby(['sentence', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end','syn_id_head', 'syn_id_tail']).agg(list)
    df = df.groupby(['syn_id_head', 'syn_id_tail','sentence']).agg(list)

    new_link_name_column = [list(set(itertools.chain(*list_of_iterables))) for list_of_iterables in df.loc[:,"link_name"]]
    df = df.map(lambda x: x[0])
    df["link_name"] = new_link_name_column
    df.reset_index(inplace=True,drop=False)
    
   
    df = df[['sentence', 'syn_id_head', 'syn_id_tail', 'link_name']]
    
    print("Dataset loaded")
    
    if relation_dict is None:
        codes, relation_dict = pd.factorize(df['link_name'].explode())
        relation_dict = dict([(el, i) for i, el in enumerate(relation_dict)])
    else:
        # Check if all values in df['link_name'] are in relation_dict
        if all(value in relation_dict for value in df['link_name'].explode().unique()):
            # If all values exist in relation_dict, proceed with the mapping
            pass
        else:
            # Remove rows where at least one value is not in relation_dict
            df = df[df['link_name'].apply(lambda x: all(i in relation_dict for i in x))]
    
    print("Relations mapped")

    df['HT'] = df['syn_id_head'] + '#' + df['syn_id_tail']

    df=df[['sentence','HT','link_name']]
    df.rename(columns={'sentence':'text','HT':'HT','link_name':'Rels'}, inplace=True)
    df['Rels'] = df['Rels'].apply(set_minimal_lists)

    
    
    return relation_dict, df