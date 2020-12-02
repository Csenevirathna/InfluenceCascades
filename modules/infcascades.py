'''
This code is authored by Chathurani Senevirathna

Also this is an implementaion for the presentaiton (oral) at the 5th IC2S2 conference, university of Amsterdam, 2019 
(https://2019.ic2s2.org/oral-presentations/) and the paper submission for the journal Entropy.

This has a set of functions to visualize the Influence Cascades
'''

import pandas as pd
import numpy as np
import networkx as nx
import plotly 
import plotly.plotly as py
from . import initialization as init
from . import globalvars as gb


# platforms = ["GitHub","Twitter"]
# communities = ["Crypto","CVE"]

platforms = gb.platforms
communities = gb.communities


def CreateSankeyLinksAndNodes_dfs(flow_df):   
    '''
    Get links and nodes dataframes to create sankey plot using the median influence values of each influence type by level.
    Will return sankey_links_df and sankey_nodes_df.
    '''
    ############## link data to create sankey plot ###################
    sankey_links_df = flow_df.melt(id_vars=['level'])
    sankey_links_df['from_label'] = sankey_links_df.apply(lambda x: "C" if "contributionTo" in x.variable else ("S" if "sharingTo" in x.variable else "I"),axis=1)
    sankey_links_df['to_label'] = sankey_links_df.apply(lambda x: "C" if "Tocontribution" in x.variable else ("S" if "Tosharing" in x.variable else "I"),axis=1)
    sankey_links_df['from_id_temp'] = (sankey_links_df.level-1).astype(str) + sankey_links_df.from_label
    sankey_links_df['to_id_temp'] = sankey_links_df.level.astype(str) + sankey_links_df.to_label
   
    ###### Reassign from_id_temp and to_id_temp with integers. To create sankey plot, nodes ids should be integers######
    node_id=pd.DataFrame(columns=['node_id_temp'])
    node_id['node_id_temp']=sankey_links_df['from_id_temp'].append(sankey_links_df['to_id_temp']).reset_index(drop=True)
    node_id=node_id.drop_duplicates('node_id_temp',keep='first')
    node_id['node_id']=np.arange(len(node_id))
    
    sankey_links_df = pd.merge(sankey_links_df, node_id, how = 'left', left_on='from_id_temp', right_on='node_id_temp',sort = False ).drop(['node_id_temp', 'from_id_temp'], axis = 1).rename(columns = {'node_id': 'from_id'})   #reassign the temp_from_id 
    sankey_links_df = pd.merge(sankey_links_df, node_id, how = 'left', left_on='to_id_temp', right_on='node_id_temp',sort = False ).drop(['node_id_temp','to_id_temp'], axis = 1).rename(columns = {'node_id': 'to_id'}) #reassign the temp_to_id
    sankey_links_df['link_color'] = np.where((sankey_links_df.from_label == "C"), "rgb(144,237,244)", np.where((sankey_links_df.from_label == "I"),"rgb(247,203,81)", "rgb(225,174,242)"))
    # sankey_links_df.to_csv("output/sankey_links_df.csv")

    ################## node data to create sankey plot #################
    sankey_nodes_df=node_id.copy()
    sankey_nodes_df['node_label']=sankey_nodes_df['node_id_temp'].apply(lambda x: "C" if "C" in x else ("I" if "I" in x else ("S" if "S" in x else "Nan")))
    sankey_nodes_df['node_color'] = np.where((sankey_nodes_df.node_label == "C"),"rgb(60,222,234)",np.where((sankey_nodes_df.node_label == "I"),"rgb(232,176,25)", "rgb(213,128,242)"))
    sankey_nodes_df=sankey_nodes_df.drop(['node_id_temp'], axis=1)
    # sankey_nodes_df.to_csv("output/sankey_nodes_df.csv")

    return sankey_links_df,sankey_nodes_df


def plotSankey(sankey_links_df,sankey_nodes_df):
    '''
    plotting sankey graph
    '''
    
    #plotly.tools.set_credentials_file(username='username', api_key='api_key') 
    data = dict(
        type='sankey',
        arrangement = "freeform",
        opacity = 0.1,
        showlegend = True,
        textfont = dict (
            color = "black",
            size = 1, 
            family = "Droid Serif"      
        ),
        domain = dict(
          x =  [0,1],
          y =  [0,1]
        ),
        node = dict(
          pad = 15,
          thickness = 30,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = sankey_nodes_df['node_label'].dropna(axis=0, how='any'),
          color = sankey_nodes_df['node_color'].dropna(axis=0, how='any')
        ),
        link = dict(
          source = sankey_links_df['from_id'],
          target = sankey_links_df['to_id'],
          value = sankey_links_df['value'],
          color = sankey_links_df['link_color'].dropna(axis=0, how='any')
      ))

    layout =  dict(
        autosize = False,
        height = 950,
        width = 1500,
        font = dict(
          size = 10
        )
    )

    fig = dict(data=[data], layout=layout)
    py.plot(fig)


def PlottingInfluenceCascades(empirical_rootchild_df,scalefree_rootchild_df,empirical_inflcascades_df,scalefree_inflcascades_df):
    '''Plotting influence cascades (Sankey plots)'''
    for platform in platforms:
        for community in communities:
            ####Empirical influence cascades                        
            inflcascades_df = empirical_inflcascades_df[(empirical_inflcascades_df["platform"]==platform) & 
                                                        (empirical_inflcascades_df["community"]==community)]
            gt_infl_medians_by_level_and_type_df = inflcascades_df.groupby(by=["level"]).median().fillna(0).reset_index() #taking the median of noramlized total influence vector components 
            sankey_links_df,sankey_nodes_df = CreateSankeyLinksAndNodes_dfs(gt_infl_medians_by_level_and_type_df) # creating nodes and flow(links) dataframes for visualization 
            plotSankey(sankey_links_df,sankey_nodes_df) # plotting influence cascades through Sankey diagram
            print("empirical", platform, community, "is plotted")
            ####Scale-free influence cascades            
            inflcascades_df_SF = scalefree_inflcascades_df[(scalefree_inflcascades_df["platform"]==platform) & 
                                                        (scalefree_inflcascades_df["community"]==community)]
            SF_infl_medians_by_level_and_type_df = inflcascades_df_SF.groupby(by=["level"]).median().fillna(0).reset_index() #taking the median of noramlized total influence vector components 
            SF_sankey_links_df,SF_sankey_nodes_df = CreateSankeyLinksAndNodes_dfs(SF_infl_medians_by_level_and_type_df) # creating nodes and flow(links) dataframes for visualization
            plotSankey(SF_sankey_links_df,SF_sankey_nodes_df) # plotting influence cascades through Sankey diagram
            print("sacle-free", platform, community, "is plotted")


def GeneratingExampleScalefreeNetworks(): 
    ''' Generating example scale-free null models'''
    
    print("Generating example scale-free null models")
    #networksizes = [50,100,200,300,400,510,513,514,515,600,800,1000]
    networksizes = [50,510,1000]
    for n in networksizes:
        print(f'network size:{n}')
        scalefree_inflnetwork = init.generateScaleFreeNetworks(n) #generating scale-free networks with equal number of nodes in empirical network
        G_SF = nx.from_pandas_edgelist(scalefree_inflnetwork,'userID0','userID1',scalefree_inflnetwork.columns.tolist()[2:],
                                   create_using=nx.DiGraph())
        indeg_zero_users_SF = init.indegZeroUsers(G_SF) # finding in-degree zero nodes (source nodes) in the influence network
        inflcascades_df_SF, root_child_df_SF = init.influenceCascadeExtraction(indeg_zero_users_SF,G_SF) # extracting influence cascades of source nodes
        SF_infl_medians_by_level_and_type_df = inflcascades_df_SF.groupby(by=["level"]).median().fillna(0).reset_index() #taking the median of noramlized total influence vector components 
        SF_sankey_links_df,SF_sankey_nodes_df = CreateSankeyLinksAndNodes_dfs(SF_infl_medians_by_level_and_type_df) # creating nodes and flow(links) dataframes for visualization
        plotSankey(SF_sankey_links_df,SF_sankey_nodes_df)