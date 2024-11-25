# Lets Go!
import Stemmer
import numpy as np
import logging
import re
from pprint import pprint
import json
import math
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
logging.getLogger('gensim').setLevel(logging.WARNING)

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

from gensim.models import LdaModel

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING)
    format="{} : %(asctime)s - %(levelname)s : %(message)s".format("Text Analysis Module") # Log message format
)

import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Call the wrapped function
        end_time = time.time()  # End the timer
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Create a logger instance
logger=logging.getLogger()

def text_cleaner(text):
    """
    Cleans the input text by removing all special characters, performing case folding and replacing - with space. Also removes extra spaces

    Args:
        text (str): The input text to be cleaned

    Returns:
        str: The cleaned text
    """
    # cleaned_text=re.sub(r"[^a-zA-Z0-9\s-]", '',text).lower().replace("\n",' ').replace("  "," ")
    cleaned_text=re.sub(r"[^a-zA-Z0-9\s-]", '',text).lower().replace("\n",' ').replace("  "," ").replace('-',' ')
    cleaned_text=re.sub(' +', ' ',cleaned_text)
    return cleaned_text

def text_tokenizer(text):
    """
    Tokenizes the input text by splitting it into list of words

    Args:
        text (str): The input string to be tokenized

    Returns:
        list: list of words obtained by splitting the input text
    """
    return text.split()

def stopword_remover(text,stop_word_path):
    """
        Removes stopwords from the given text

        Args:
            text (list of str): The input text represented as a list of words
            stop_word_path (str): stopwords file path

        Returns:
            list of str: The text with stopwords removed
    """
    file=open(stop_word_path,'r')
    stop_word_set=set(file.read().split())
    file.close()
    return [word for word in text if word not in stop_word_set]

def text_stemmer(text,lang='english'):
    """
        Stems the words in the given text using the passed stemming algorithm

        Args:
            text (list of str) list of words to be stemmed
            lang (str, optional): The stemming algorithm to use (porter default)

        Returns:
            list of str list of stemmed text
    """
    ps=Stemmer.Stemmer(lang)
    return [ps.stemWord(word) for word in text]

@timeit
def parse_and_prepare_file(file_path):
    '''
    Read file and create:
    {
    class1:
    {doc_list:
        [
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]},
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]},
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]}
        ],}
    class2:
    {doc_list:
        [
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]},
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]},
        {doc_id:xxxx, text:[tokenised and pre-processed doc text]}
        ]}

    }
    '''
    logging.info(f"Starting to read and process file:{file_path.split('/')[-1]}")
    data={}
    with open(file_path,'r') as file:
        for line in file:
            part_line=line.split('\t')
            clean_text=text_stemmer(stopword_remover(text_tokenizer(text_cleaner(part_line[1])),"./data/input/english_stop_list.txt"))
            class_name=part_line[0]

            if class_name not in data:
                data[class_name]={'doc_list':[]}
                last_doc=0
            else:
                last_doc=len(data[class_name]['doc_list'])
            
            data[class_name]['doc_list'].append({'doc_id': last_doc+1,
                                     'text': clean_text})

    return data
        
@timeit 
def create_inverted_index(data):
    '''
    before:
    Read file and create:
    {'OT': {'doc_list': [{'doc_id': 1,
                      'text': ['begin', 'god', 'creat', 'heaven', 'earth']},
                     {'doc_id': 2,
                      'text': ['earth',
                               'form',
                               'water']},
                     {'doc_id': 3, 'text': ['god', 'light', 'light']},
                     {'doc_id': 4,
                      'text': ['god',
                               'light',
                               'good']}
            },
        'NT':.....

    index struct:

    index={
    OT:{
    word_count: 32432,
    doc_count: 324,
    terms:
    {
    term: 'abc',
    document_freq:23,
    doc_list:[4,5,6,34,65,43,3]

    }


    '''
    logging.info("Starting Index creation")
    global_index={}
    for class_name in data:
        index={}
        for doc in data[class_name]['doc_list']:
            for word in doc['text']:
                if word not in index:
                    index[word]={
                        'df':1,
                        'doc_list':[doc['doc_id']]
                    }
                else:
                    if doc['doc_id'] in index[word]['doc_list']:
                        pass
                    else:
                        index[word]['doc_list'].append(doc['doc_id'])
                        index[word]['df']+=1
        global_index[class_name]={
            'word_count':len(index.keys()),
            'doc_count':len(data[class_name]['doc_list']),
            'term_index':index
        }
        logging.info(f"Index creation complete for {class_name}")
    
    # pprint(global_index)
    with open('./data/input/index.json', 'w') as fp:
        json.dump(global_index, fp, indent=4)
    logging.info("Index file written to path")
    return global_index

@timeit
def calculate_MI(index):
    '''
    MI;
    N00: No. of documents where term t is not present and class c is also not present
    ((total docs-docs in class)- no of docs where term t present in other 2 classes)
    N01: No. of docs where term t not present but document belongs to class c
    (Total docs in class - documents with term t)
    N10: no of docs where term t is present but document does not belong to class c
    (no. of docs where term is present in other two classes)
    N11: No of terms where document has term t and belongs to class c
    (doc freq of term in class c)
    N: total docs
    '''
    class_set=index.keys()
    # print(class_set)
    MI={}
    for class_name in index:
        MI[class_name]={}
        for word in index[class_name]['term_index']:
            # print(word)
            N00=0
            N10=0
            for classes in class_set:
                if class_name!=classes:
                    N00=N00+index[classes]['doc_count']
                    N00=N00-(index[classes]['term_index'][word]['df'] if word in index[classes]['term_index'] else 0)
                    N10=N10+(index[classes]['term_index'][word]['df'] if word in index[classes]['term_index'] else 0)
            # print(class_name)
            # print(word)
            # print(N00)
            N01=index[class_name]['doc_count']-index[class_name]['term_index'][word]['df']
            # print(N01)
            # print(N10)
            N11=index[class_name]['term_index'][word]['df']
            # print(N11)

            N=sum([index[cls]['doc_count'] for cls in class_set])
            MI_score=0
            if N11>0:
                MI_score=((N11/N) * math.log2((N*N11)/((N10+N11)*(N01+N11))))
            if N01>0:
                MI_score+=((N01/N) * math.log2((N*N01)/((N00+N01)*(N11+N01))))
            if N10>0:    
                MI_score+=((N10/N) * math.log2((N*N10)/((N10+N11)*(N00+N10))))
            if N00>0:    
                MI_score+=((N00/N) * math.log2((N*N00)/((N01+N00)*(N10+N00))))
            MI[class_name][word]=MI_score
        
    return MI




@timeit
def calculate_X2(index):
    '''
    MI;
    N00: No. of documents where term t is not present and class c is also not present
    ((total docs-docs in class)- no of docs where term t present in other 2 classes)
    N01: No. of docs where term t not present but document belongs to class c
    (Total docs in class - documents with term t)
    N10: no of docs where term t is present but document does not belong to class c
    (no. of docs where term is present in other two classes)
    N11: No of terms where document has term t and belongs to class c
    (doc freq of term in class c)
    N: total docs
    '''
    class_set=index.keys()
    # print(class_set)
    X2={}
    for class_name in index:
        X2[class_name]={}
        for word in index[class_name]['term_index']:
            # print(word)
            N00=0
            N10=0
            for classes in class_set:
                if class_name!=classes:
                    N00=N00+index[classes]['doc_count']
                    N00=N00-(index[classes]['term_index'][word]['df'] if word in index[classes]['term_index'] else 0)
                    N10=N10+(index[classes]['term_index'][word]['df'] if word in index[classes]['term_index'] else 0)
            # print(class_name)
            # print(word)
            # print(N00)
            N01=index[class_name]['doc_count']-index[class_name]['term_index'][word]['df']
            # print(N01)
            # print(N10)
            N11=index[class_name]['term_index'][word]['df']
            # print(N11)

            N=sum([index[cls]['doc_count'] for cls in class_set])


            X2_score= (N11+N10+N01+N00) * math.pow(((N11*N00)-(N10*N01)),2)
            X2_score/= (N11+N01)*(N11+N10)*(N10+N00)*(N01+N00)
            
            X2[class_name][word]=X2_score
        
    return X2
            

@timeit
def topic_modelling(path):
    data={}
    with open(file_path,'r') as file:
        with open(file_path,'r') as file:
            for line in file:
                part_line=line.split('\t')
                clean_text=text_stemmer(stopword_remover(text_tokenizer(text_cleaner(part_line[1])),"./data/input/english_stop_list.txt"))
                class_name=part_line[0]
                if class_name not in data:
                    data[class_name]=[]
                
                data[class_name].append(clean_text)
    
    logger.info("Data Parsed for all three classes")

    common_texts=[]
    for class_name in data:
        for text in data[class_name]:
            common_texts.append(text)
    # print(len(common_texts))
    # with open('res.json', 'w') as fp:
    #     json.dump(common_texts, fp, indent=4)

    # Create a corpus from a list of texts
    common_dictionary = Dictionary(common_texts)

    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

    # Train the model on the corpus.
    # lda = LdaModel(common_corpus, num_topics=20)
    # lda.save("lda_model_k20.gensim")

    lda = LdaModel.load("lda_model_k20.gensim")
    print("Model successfully loaded")


    # topics = lda.print_topics(num_topics=20, num_words=10)
    # for topic_id, topic_words in topics:
    #     print(f"Topic {topic_id}: {topic_words}")
    

    # lda_vis = gensimvis.prepare(lda, common_corpus, common_dictionary)
    # pyLDAvis.save_html(lda_vis, 'lda_visualization.html')

    # pyLDAvis.show(lda_vis)

    # Calculating doc-topic probabilities
    class_topic_scores = {}
    
    for class_name in data:
        corpus_dictionary=Dictionary(data[class_name])
        corpus= [corpus_dictionary.doc2bow(text) for text in data[class_name]]

        topic_scores =[0]*20

        for doc_bow in corpus:
            doc_topics = lda.get_document_topics(doc_bow, minimum_probability=0.0)
            for topic_id,prob in doc_topics:
                topic_scores[topic_id]+=prob

        avg_topic_scores = [score / len(corpus) for score in topic_scores]

        top_topic_id=int(np.argmax(avg_topic_scores))
        top_score=avg_topic_scores[top_topic_id]
                                   
        top_10_tokens = lda.show_topic(top_topic_id, topn=10)
        top_10_words = [(common_dictionary[int(word_id)], prob) for word_id, prob in top_10_tokens]
        

        class_topic_scores[class_name]={
            "top_topic_id":top_topic_id,
            "top_avg_score":top_score,
            "token_list": top_10_tokens,
            "word_lists": top_10_words
        }
    pprint(class_topic_scores)
        

        
        
           
    



            

            
        




if __name__ == "__main__":
    file_path="./data/input/bible_and_quran.tsv"

    # data=parse_and_prepare_file(file_path)
    # pprint(data)
    # g_index=create_inverted_index(data)

    with open("./data/input/index.json", "r") as json_file:
        index = json.load(json_file)

    # for class_name in index:
    #     print(class_name)
    #     print(index[class_name]['word_count'])
    #     print(index[class_name]['doc_count'])
    
    # all_terms_MI=calculate_MI(index)
    # top_10_MI={}
    # for class_name in all_terms_MI:
    #     top_10_MI[class_name]=sorted(all_terms_MI[class_name].items(), key=lambda item: item[1], reverse=True)[:10]
    # pprint(top_10_MI)

    # all_terms_X2=calculate_X2(index)
    # top_10_X2={}
    # for class_name in all_terms_X2:
    #     top_10_X2[class_name]=sorted(all_terms_X2[class_name].items(), key=lambda item: item[1], reverse=True)[:10]
    # pprint(top_10_X2)


    res=topic_modelling(file_path)


    




    