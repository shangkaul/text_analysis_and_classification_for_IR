# Evaluate IR Results
import pandas as pd
import logging
import numpy as np

# input files in data folder:
# system_results.csv: a file containing the retrieval results of a given IR system and
# qrels.csv: a file that has the list of relevant documents for each of the queries.


# Configure logging to display messages in the console (stdout)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING)
    format="{} : %(asctime)s - %(levelname)s : %(message)s".format("IR EVAL Module") # Log message format
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
@timeit
def read_csv(file_path):
    df=pd.read_csv(file_path)
    logging.info(f"CSV file {file_path.split('/')[-1]} read successfully")
    return df


@timeit
def p_10(qrels,sys_res):
    '''
    Precision at 10: Out of top 10 retrieved docs how many were relevant
    -> Relevant,retrieved / all retrieved
    '''
    
    result=[]
    system_list=sys_res['system_number'].unique()
    query_list= sys_res['query_number'].unique()

    for sys in system_list:
        sys_p10=[]
        for query in query_list:
            query_res_df= sys_res[(sys_res['system_number'] == sys) & (sys_res['query_number'] == query)]
            
            df_rank_10 = query_res_df.sort_values('rank_of_doc')
            retrieved_docs=df_rank_10['doc_number'].head(10).tolist()
            # print(retrieved_docs)

            relevant_docs=qrels[qrels['query_id']==query]['doc_id'].tolist()
            # print(relevant_docs)

            relevant_retrieved_docs=list(set(relevant_docs) & set(retrieved_docs))

            # print(relevant_retrieved_docs)

            p_score=len(relevant_retrieved_docs)/10
            sys_p10.append(p_score)
            
            result.append({'system_number':int(sys),'query_number':int(query),'P@10':round(float(p_score),3)})
            # break
        mean_p10=sum(sys_p10)/len(sys_p10) if sys_p10 else 0
        result.append({'system_number':int(sys),'query_number':'mean','P@10':round(float(mean_p10),3)})

        # break
    
    res_df= pd.DataFrame(result)
    return res_df


@timeit
def r_50(qrels,sys_res):
    '''
    r50=R@50 -> No. of relevant docs retrieved(50 cutoff)/total relevant docs
    '''
    result=[]
    system_list=sys_res['system_number'].unique()
    query_list= sys_res['query_number'].unique()

    for sys in system_list:
        sys_r50=[]
        for query in query_list:
            query_res_df= sys_res[(sys_res['system_number'] == sys) & (sys_res['query_number'] == query)]
            
            relevant_docs=qrels[qrels['query_id']==query]['doc_id'].tolist()
            
            df_rank_50 = query_res_df.sort_values('rank_of_doc')
            retrieved_docs=df_rank_50['doc_number'].head(50).tolist()

            relevant_retrieved_docs=list(set(relevant_docs) & set(retrieved_docs))

            r_score=len(relevant_retrieved_docs)/len(relevant_docs)
            sys_r50.append(r_score)

            result.append({'system_number':int(sys),'query_number':int(query),'R@50':round(float(r_score),3)})
        mean_r50=sum(sys_r50)/len(sys_r50)
        result.append({'system_number':int(sys),'query_number':'mean','R@50':round(float(mean_r50),3)})

        # print(result)
    res_df= pd.DataFrame(result)
    return res_df

@timeit
def rp(qrels,sys_res):
    '''
    rPrecision -> precision at R =>relevant, retrieved docs in top R/R (R=num of relevant docs)
    '''
    result=[]
    system_list=sys_res['system_number'].unique()
    query_list= sys_res['query_number'].unique()

    for sys in system_list:
        sys_rp=[]
        for query in query_list:
            query_res_df= sys_res[(sys_res['system_number'] == sys) & (sys_res['query_number'] == query)]
            
            relevant_docs=qrels[qrels['query_id']==query]['doc_id'].tolist()

            n=len(relevant_docs)
            
            df_rank_n = query_res_df.sort_values('rank_of_doc')
            retrieved_docs=df_rank_n['doc_number'].head(n).tolist()

            relevant_retrieved_docs=list(set(relevant_docs) & set(retrieved_docs))

            rp_score=len(relevant_retrieved_docs)/n
            sys_rp.append(rp_score)

            result.append({'system_number':int(sys),'query_number':int(query),'r-precision':round(float(rp_score),3)})
        mean_rp=sum(sys_rp)/len(sys_rp)
        result.append({'system_number':int(sys),'query_number':'mean','r-precision':round(float(mean_rp),3)})

        # print(result)
    res_df= pd.DataFrame(result)
    return res_df


@timeit
def ap(qrels,sys_res):
    '''
    Avg Precision = At each match, check rank. nmatch/rank -> do this till all relevant docs found. 
    then take average of all.
    '''
    result=[]
    system_list=sys_res['system_number'].unique()
    query_list= sys_res['query_number'].unique()

    for sys in system_list:
        sys_ap=[]
        for query in query_list:
            query_res_df= sys_res[(sys_res['system_number'] == sys) & (sys_res['query_number'] == query)]
            
            relevant_docs=qrels[qrels['query_id']==query]['doc_id'].tolist()

            n=len(relevant_docs)
            
            df_rank_n = query_res_df.sort_values('rank_of_doc')
            retrieved_docs=df_rank_n['doc_number'].tolist()

            relevant_count=0
            matches=[]
            rank=1
            for doc in retrieved_docs:
                if doc in relevant_docs:
                    relevant_count=relevant_count+1
                    matches.append(relevant_count/rank)
                rank=rank+1
            

            ap_score=sum(matches)/n
            sys_ap.append(ap_score)

            result.append({'system_number':int(sys),'query_number':int(query),'AP':round(float(ap_score),3)})
        mean_ap=sum(sys_ap)/len(sys_ap)
        result.append({'system_number':int(sys),'query_number':'mean','AP':round(float(mean_ap),3)})

        # print(result)
    res_df= pd.DataFrame(result)
    return res_df


@timeit
def ndcg(qrels,sys_res,cutoff):
    '''

    '''
    result=[]
    system_list=sys_res['system_number'].unique()
    query_list= sys_res['query_number'].unique()
    # system_list=[5]
    # query_list=[8]
    
    for sys in system_list:
        sys_dcg=[]
        for query in query_list:
            query_res_df= sys_res[(sys_res['system_number'] == sys) & (sys_res['query_number'] == query)]
            query_res_df=query_res_df.drop('score', axis=1).sort_values('rank_of_doc')
            # print(query_res_df)

            # print(qrels)

            joined_df= pd.merge(query_res_df,qrels, left_on=['query_number','doc_number'],
                                                    right_on=['query_id','doc_id'],
                                                    how='left')
            
            joined_df=joined_df.drop(['query_id','doc_id'],axis=1)

            joined_df['relevance']=joined_df['relevance'].fillna(0).astype(int)
            
            # print(joined_df)

            # joined_df.to_csv('./data/output/test.csv', index=False, mode='w')
            
            dcg_df=joined_df.head(cutoff).copy()
            dcg_df['dg']= dcg_df['relevance']/np.log2(dcg_df['rank_of_doc'])
            dcg_df['dg'] = dcg_df['dg'].replace([np.inf, -np.inf], np.nan).fillna(dcg_df['relevance'])

            dcg_df['dcg_k']=dcg_df['dg'].cumsum()


            qrels_df=qrels[qrels['query_id']==query].drop(['doc_id'],axis=1)
            # print(qrels_df)
            qrels_df['relevance']=qrels_df['relevance'].sort_values(ascending=False).values

            if len(qrels_df)<cutoff:
                padding_needed = cutoff-len(qrels_df)
                padding = pd.DataFrame({"query_id": [query] * padding_needed,
                                        "relevance": [0] * padding_needed
                                       })
            
                qrels_df = pd.concat([qrels_df, padding], ignore_index=True)
            
            qrels_df=qrels_df.head(cutoff).drop(['query_id'],axis=1).reset_index(drop=True)
            qrels_df.rename(columns={'relevance': 'iG'}, inplace=True)
            # print(qrels_df)
            ndcg_df = dcg_df.join(qrels_df) #join on index
            
            ndcg_df['iG']=ndcg_df['iG'].sort_values(ascending=False).values

            # print(ndcg_df)

            # ndcg_df['iG']=ndcg_df['relevance'].sort_values(ascending=False).values
            ndcg_df['iDG']=ndcg_df['iG']/np.log2(ndcg_df['rank_of_doc'])
            ndcg_df['iDG']=ndcg_df['iDG'].replace([np.inf, -np.inf], np.nan).fillna(ndcg_df['iG'])
            ndcg_df['iDCG_k']=ndcg_df['iDG'].cumsum()

            ndcg_df['nDCG_k']=ndcg_df['dcg_k']/ndcg_df['iDCG_k']

            ndcg_df['nDCG_k']=ndcg_df['nDCG_k'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # print(ndcg_df)

            ndcg_score = round(float(ndcg_df['nDCG_k'].iloc[-1]), 3)
            
            sys_dcg.append(ndcg_score)
            result.append({'system_number':int(sys),'query_number':int(query),f'nDCG@{cutoff}':round(float(ndcg_score),3)})
            # print(result)
            # break
        mean_ndcg_score= sum(sys_dcg)/len(sys_dcg)
        result.append({'system_number':int(sys),'query_number':'mean',f'nDCG@{cutoff}':round(float(mean_ndcg_score),3)})

        # print(result)
        # break
    res_df= pd.DataFrame(result)
    return res_df

        
            


@timeit
def EVAL(qrels,sys_res):
    res_p10=p_10(qrels,sys_res)
    # print(res_p10)

    res_r50=r_50(qrels,sys_res)

    res_df=pd.merge(res_p10,res_r50,on=['system_number','query_number'])

    res_rp=rp(qrels,sys_res)

    res_df=pd.merge(res_df,res_rp,on=['system_number','query_number'])

    res_ap=ap(qrels,sys_res)
    res_df=pd.merge(res_df,res_ap,on=['system_number','query_number'])

    res_dcg_10=ndcg(qrels,sys_res,10)
    res_df=pd.merge(res_df,res_dcg_10,on=['system_number','query_number'])

    res_dcg_20=ndcg(qrels,sys_res,20)
    res_df=pd.merge(res_df,res_dcg_20,on=['system_number','query_number'])




    # print(res_df)
    res_df.to_csv('./data/output/ir_eval.csv', index=False, mode='w')


    


if __name__ == "__main__":
    # Add your main code here
    query_results=read_csv("./data/input/qrels.csv")
    # print(query_results.head())
    system_results=read_csv("./data/input/system_results.csv")
    # print(system_results.head())

    EVAL(query_results,system_results)
