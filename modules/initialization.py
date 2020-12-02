'''
This code is authored by Chathurani Senevirathna

Also this is an implementaion for the presentaiton (oral) at the 5th IC2S2 conference, university of Amsterdam, 2019 
(https://2019.ic2s2.org/oral-presentations/) and the paper submission for the journal Entropy.

This has set of functions to extract Influence Cascades and to create data frames for analysis.
    
'''

import pandas as pd
import numpy as np
import networkx as nx
import random
import decimal
import glob
from collections import OrderedDict
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from . import globalvars as gb

# relationshipTypes=['creationTocreation', 'creationTocontribution', 'creationTosharing', 
#                    'contributionTocreation', 'contributionTocontribution', 'contributionTosharing', 
#                    'sharingTocreation', 'sharingTocontribution', 'sharingTosharing']
# platforms = ["GitHub","Twitter"]
# communities = ["Crypto","CVE"]

relationshipTypes= gb.relationshipTypes
platforms = gb.platforms
communities = gb.communities


def indegZeroUsers(G):
    '''
    Select users with indegree zero
    
    '''
    indegrees=list(G.in_degree(G.nodes()))
    indegree_zero_users = [indegree[0] for indegree in indegrees if indegree[1] == 0 ]
    print (f'number of indeg zero users {len(indegree_zero_users)}')
    return indegree_zero_users


def influenceCascadeExtraction(root_list,G):
    '''
    Extracting influence cascades rooted from a given root users list.
    
    '''   

    influence_df = pd.DataFrame()
    root_child_df = pd.DataFrame()

    for root in root_list: 

        data=[] # a list contains the influence vectors of users in each levels of hierarchy 
        Queue=[] # a list wich store nodes tempararily 
        hopdists={} # distance from root to nodes. key: node, value: distance from root

        Queue.insert(0,root)    
        hopdists[root] = 0

        while len(Queue) > 0:
            currentNode = Queue.pop()
            hopdist = hopdists[currentNode]
            outEdges = G.out_edges(currentNode,True)
            for edge in outEdges:
                influenceVector= [edge[2]['creationTocreation'], edge[2]['creationTocontribution'], 
                                  edge[2]['creationTosharing'],edge[2]['contributionTocreation'],
                                  edge[2]['contributionTocontribution'],edge[2]['contributionTosharing'],
                                  edge[2]['sharingTocreation'],edge[2]['sharingTocontribution'],
                                  edge[2]['sharingTosharing']]
                child = edge[1]
                if not child in hopdists.keys(): # check whether the node is already counted
                    if len(data) < (hopdist+1):  
                        data.append([])          # create sublists in data list for each hop
                    # store influence vectors of nodes in their corresponding sublist in data list
                    data[hopdist].append(influenceVector) 
                    # add the child node to Queue to consider it as a parent node in the next rounds
                    Queue.insert(0,child)                 
                    hopdists[child] = hopdist + 1   
        
        ##### Characterization of influence vector components #####
        levelInfluenceValues=[zip(*hop) for hop in data]    
        for i in range(len(levelInfluenceValues)):
            # total influence by activity type at each level of the cascade 
            levelInfluenceValues[i] = [sum(j) for j in levelInfluenceValues[i]]  
        
        levels = list(range(1, len(levelInfluenceValues)+1))
        levelInfluenceValues = np.array(levelInfluenceValues).T # total linfluence values at each level by activity type
        
        for idx, l in enumerate(levelInfluenceValues):
            m = np.sum(l)
            levelInfluenceValues[idx] = np.array(l) / m  # normalized total influence at each level by activity type   

        this_root_influence_by_levels = {}
        for idx, relation in enumerate(relationshipTypes):
            this_root_influence_by_levels[relation]=levelInfluenceValues[idx]
        this_root_influence_by_levels = pd.DataFrame(this_root_influence_by_levels)
    
        this_root_influence_by_levels["level"]=levels 
        this_root_influence_by_levels["root"]=root
        influence_df = influence_df.append(this_root_influence_by_levels,ignore_index=True)  
        
        ##### node-level data extraction of cascades for statistical analysis #######
        cascade_node_data = {'node':list(dict(hopdists).keys()), 'node_level':list(dict(hopdists).values())}
        root_child_by_levels = pd.DataFrame(cascade_node_data)
        root_child_by_levels["root"] = root
        root_child_df = root_child_df.append(root_child_by_levels,ignore_index = True) 

    print('influenceCascades_df')
    print (influence_df.head())
    return influence_df, root_child_df


def generateScaleFreeNetworks(n):
    '''
    Generate scale-free networks with n number of nodes
    '''
    G_scaleFree = nx.scale_free_graph(n, seed=30)# scale free graph with default probabilities (alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, create_using=None,)

    np.random.seed(0)
    def setDecimals(x):
        decimal.getcontext().prec = 10  # 10 decimal points enough
        return decimal.Decimal(0) + decimal.Decimal(x)
    # this should include both boundaries as float gets close enough to 1 to make decimal round

    edges = list(set(G_scaleFree.edges()))
    scalefree_network = pd.DataFrame(edges, columns = ["userID1","userID0" ], dtype= str)
    scalefree_network = scalefree_network[scalefree_network["userID0"]!= scalefree_network["userID1"]]
    for action in relationshipTypes:
        scalefree_network[action] = np.random.uniform(0,1, scalefree_network.shape[0])
        scalefree_network[action] = scalefree_network[action].apply(lambda x: setDecimals(x)).astype(float) 
    scalefree_inflnetwork = scalefree_network[scalefree_network.iloc[:,2:].sum(axis=1) > 0]
    return scalefree_inflnetwork


def CreateDataFrames():
    '''
    Create relevent data frames
    Returns;
    1)empirical_inflcascades_df, 2)empirical_rootchild_df,
    3)scalefree_inflcascades_df, 4)scalefree_rootchild_df
    '''

    empirical_inflcascades_df = pd.DataFrame()
    empirical_rootchild_df = pd.DataFrame()
    empirical_inflnetworks_nodecount = OrderedDict()
    scalefree_inflcascades_df = pd.DataFrame()
    scalefree_rootchild_df = pd.DataFrame()

    for platform in platforms:
        for community in communities:
            platform_community = str(platform)+"_"+str(community)
            inflnetwork_gt=pd.read_csv("input_data/"+platform_community+"_Influence_Network_df.csv")
            print (platform_community)
            G=nx.from_pandas_edgelist(inflnetwork_gt,'userID0','userID1',inflnetwork_gt.columns.tolist()[2:],
                                       create_using=nx.DiGraph())
            empirical_inflnetworks_nodecount[platform_community] = G.number_of_nodes()
            indeg_zero_users = indegZeroUsers(G) # finding in-degree zero nodes (source nodes) in the influence network
            inflcascades_df, root_child_df = influenceCascadeExtraction(indeg_zero_users,G) # extracting influence cascades of source nodes
    
            #### storing influence cascades dfs and root_child dfs for statistical analysis 
            inflcascades_df["platform"] = platform
            inflcascades_df["community"] = community
            empirical_inflcascades_df = empirical_inflcascades_df.append(inflcascades_df)
            root_child_df["platform"] = platform
            root_child_df["community"] = community
            root_child_df["platform_community"] = platform_community
            empirical_rootchild_df = empirical_rootchild_df.append(root_child_df)

            #### comparison with scale-free networks ######
            print ("SF_"+platform_community)
            scalefree_inflnetwork = generateScaleFreeNetworks(G.number_of_nodes()) #generating scale-free networks with equal number of nodes in empirical network
            G_SF = nx.from_pandas_edgelist(scalefree_inflnetwork,'userID0','userID1',scalefree_inflnetwork.columns.tolist()[2:],
                                       create_using=nx.DiGraph())
            indeg_zero_users_SF = indegZeroUsers(G_SF) # finding in-degree zero nodes (source nodes) in the influence network
            inflcascades_df_SF, root_child_df_SF = influenceCascadeExtraction(indeg_zero_users_SF,G_SF) # extracting influence cascades of source nodes

            #### Storing influence cascades dfs of scale-free networks for statistical analysis
            inflcascades_df_SF["platform"] = platform
            inflcascades_df_SF["community"] = community
            scalefree_inflcascades_df = scalefree_inflcascades_df.append(inflcascades_df_SF)
            root_child_df_SF["platform"] = platform
            root_child_df_SF["community"] = community
            root_child_df_SF["platform_community"] = "SF_"+ str(platform_community)
            scalefree_rootchild_df = scalefree_rootchild_df.append(root_child_df_SF)
            #### Return
    return empirical_inflcascades_df,empirical_rootchild_df,scalefree_inflcascades_df,scalefree_rootchild_df 



