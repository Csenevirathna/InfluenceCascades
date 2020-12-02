'''
This code is authored by Chathurani Senevirathna

Also this is an implementaion for the paper submission for the journal Entropy.

This has set of functions to do statistical analysis of user distributions and influence distributions on Influence Cascades.
'''

import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
import matplotlib.pyplot as plt
import seaborn as sns
from . import globalvars as gb

# relationshipTypes=['creationTocreation', 'creationTocontribution', 'creationTosharing', 
#                    'contributionTocreation', 'contributionTocontribution', 'contributionTosharing', 
#                    'sharingTocreation', 'sharingTocontribution', 'sharingTosharing']
# platforms = ["GitHub","Twitter"]
# communities = ["Crypto","CVE"]

relationshipTypes= gb.relationshipTypes
platforms = gb.platforms
communities = gb.communities


def UserDistributions(empirical_rootchild_df,scalefree_rootchild_df):
    '''
    Analysis of user distributions 
    '''
    rootchild_data = pd.concat([empirical_rootchild_df,scalefree_rootchild_df], axis=0)
    user_count_at_levels = rootchild_data.groupby(["platform","community","platform_community","root","node_level"]).size().reset_index().rename(columns={0:"num_nodes"})
    avg_user_count_at_levels = user_count_at_levels.groupby(["platform","community","platform_community","node_level"]).mean().reset_index()
    avg_user_count_at_levels["network"] = avg_user_count_at_levels["platform_community"].apply(lambda x: "Scale-free" if "SF" in x else "Empirical") 
    avg_user_count_at_levels["diff"] = avg_user_count_at_levels.groupby("platform_community")["num_nodes"].diff(periods=1)
    avg_user_count_at_levels["cumulative_sum_of_users"] = avg_user_count_at_levels.groupby("platform_community")["num_nodes"].apply(lambda x: x.cumsum())
    avg_user_count_at_levels["norm_num_nodes"] = avg_user_count_at_levels.groupby("platform_community")["num_nodes"].apply(lambda x : x/x.max())
    avg_user_count_at_levels["norm_cumulative_sum_of_users"] =avg_user_count_at_levels.groupby("platform_community")["cumulative_sum_of_users"].apply(lambda x: x/x.max())
    avg_user_count_at_levels["norm_diff"] = avg_user_count_at_levels.groupby("platform_community")["diff"].apply(lambda x : x/abs(x).max())
    
    #### Mean number of users per cascade level by platform and community
    with sns.plotting_context("paper",font_scale=2):
        print("Plotting Mean number of users per cascade level by platform and community")
        h1 = sns.catplot(x="node_level", y="norm_num_nodes", row="platform",col="community",hue="network",palette=["C0","C1"],kind="bar",
                    data=avg_user_count_at_levels)
        h1.set_xlabels("Level",fontsize= 20)
        h1.set_ylabels("Total Number of Users",fontsize= 20)
        h1.set_titles('{row_name}' ' | ' '{col_name}',fontsize=20)
        h1.fig.subplots_adjust(wspace=0.1)
        h1._legend.set_title("")
    plt.show()    
      
    #### Cumulative mean number of users per cascade level by platform and community
    print("Plotting cumulative mean number of users per cascade level by platform and community")
    with plt.rc_context(dict(sns.axes_style("whitegrid",{'grid.linestyle': '--'}),
                             **sns.plotting_context("paper", font_scale=2))):
        h2=sns.relplot(x="node_level", y="norm_cumulative_sum_of_users", row="platform",col="community",hue="network",palette=["C0","C1"],
                       kind="line", data=avg_user_count_at_levels, linewidth=3)    
        h2.set_xlabels("Level",fontsize= 20)
        h2.set_ylabels("Cumulative Sum of Users",fontsize= 20)
        h2.set_titles('{row_name}' ' | ' '{col_name}',fontsize=20)
        h2.fig.subplots_adjust(wspace=0.1)
        h2._legend.texts[0].set_text("")
    plt.show()
    
    ####################### JS-divergence tests ########################################
    #### JS-divergence of mean number of nodes distribution between each empirical network
    print("JS-divergence test of mean number of nodes distribution between each empirical network")
    from scipy.spatial import distance
    jstest_df=avg_user_count_at_levels[avg_user_count_at_levels["network"]=="Empirical"]
    dist={}
    for x in list(jstest_df["platform_community"].unique()):
        for y in list(jstest_df["platform_community"].unique()):        
            df1= jstest_df[jstest_df["platform_community"]==x]
            df2= jstest_df[jstest_df["platform_community"]==y]
            m = min(df1["node_level"].max(),df2["node_level"].max())       
            df1= df1[df1["node_level"]<=m]
            df2= df2[df2["node_level"]<=m]
            dist[x,y]=distance.jensenshannon(np.array(df1["norm_num_nodes"]), np.array(df2["norm_num_nodes"]))

    df = {}
    df["platform_community_comb"]=list(dist.keys())
    df["distance"]=list(dist.values())

    df1 = pd.DataFrame(df)
    df1.drop_duplicates()
    print(df1.sort_values(by=['distance']))
    df1.sort_values(by=['distance']).to_csv("JSTestMeanUserDist.csv")
    
    #### JS-divergence of cumulative user distributions between each network
    print("JS-divergence test of cumulative user distributions between each network")
    from scipy.spatial import distance
    dist={}
    for x in list(avg_user_count_at_levels["platform_community"].unique()):
        for y in list(avg_user_count_at_levels["platform_community"].unique()):        
            df1= avg_user_count_at_levels[avg_user_count_at_levels["platform_community"]==x]
            df2= avg_user_count_at_levels[avg_user_count_at_levels["platform_community"]==y]
            m = min(df1["node_level"].max(),df2["node_level"].max())       
            df1= df1[df1["node_level"]<=m]
            df2= df2[df2["node_level"]<=m]
            dist[x,y]=distance.jensenshannon(np.array(df1["norm_cumulative_sum_of_users"]), np.array(df2["norm_cumulative_sum_of_users"]))

    df = {}
    df["platform_community_comb"]=list(dist.keys())
    df["distance"]=list(dist.values())

    df1 = pd.DataFrame(df)
    df1.drop_duplicates()
    df1.sort_values(by=['distance'])
    df1.to_csv("JSTestCumulativeUserDist.csv")
    print(df1)


def InfluenceDistribution(empirical_inflcascades_df,scalefree_inflcascades_df):
    '''
    Analysis of the residuals differences between the median normalized total influence values extracted 
    from influence cascades and those from influence cascades generated by the corresponding null model
    '''
    empirical_inflcascades_medians = empirical_inflcascades_df.groupby(by=["platform","community","level"]).median().fillna(0).reset_index() #taking the median of noramlized total influence vector components 
    scalefree_inflcascades_meidans = scalefree_inflcascades_df.groupby(by=["platform","community","level"]).median().fillna(0).reset_index()
    residual_df = empirical_inflcascades_medians.set_index(["platform","community","level"]).subtract(scalefree_inflcascades_meidans.set_index(["platform","community","level"]), fill_value=0).reset_index()
    residual_df["platform_community"]=residual_df["platform"]+"_"+residual_df["community"]
    supremum_level=min(list(residual_df.groupby(["platform_community"])["level"].max()))
    truncated_residual_df = residual_df[residual_df["level"]<=supremum_level]

    ###### Plotting residuals of median normalized total influence values #######
    print("Plotting residuals of median normalized total influence values")
    df=pd.melt(truncated_residual_df, id_vars=["level","platform_community"], value_vars=truncated_residual_df.columns[3:-1],
               var_name="inf_relationship", value_name="influence")
    df["inf_relationship"]=df["inf_relationship"].str.replace("creation","initiation") # replace the word "creation" by "initiation"
    with sns.plotting_context("paper",font_scale=1.9):
        f= sns.relplot(x="level", y="influence", col="inf_relationship", col_wrap=3, hue="platform_community", 
                       palette=["C0","C1","C2","k"], kind="line", linewidth=2, data=df, height=4.9, aspect=0.8)
        (f.map(plt.axhline, y=0, color=".7", dashes="", lw=2))

        f.set_xlabels("Level",fontsize= 20)
        f.set_ylabels("Influence",fontsize= 20)
        f.set_titles("{col_name}",fontsize=18)
        f.fig.subplots_adjust(wspace=0.2)
        leg = f._legend
        leg.texts[0].set_text("")
        f.set(xticks=df.level[0::1])    
    plt.show()


    ##### Spearman's correlation test between platforms and community by influence relationships
    #####(only first four levels are considered)
    def corrByRelationship(df1,df2):
        corr_coef_by_relationship = {}
        corr_p_by_relationship = {}
        for relation in relationshipTypes:
            corr_coef_by_relationship[relation]=scipy.stats.spearmanr(np.array(df1[relation]), np.array(df2[relation]))[0]
            corr_p_by_relationship[relation]=scipy.stats.spearmanr(np.array(df1[relation]), np.array(df2[relation]))[1]
        return corr_coef_by_relationship,corr_p_by_relationship


    spearmann_df = pd.DataFrame(columns=["inf_relationship"])
    spearmann_df["inf_relationship"]=relationshipTypes
    pd.set_option("display.precision",8)
    for platform in ["GitHub","Twitter"]:
        df1 = truncated_residual_df[truncated_residual_df["platform_community"]==platform+"_Crypto"]
        df2 = truncated_residual_df[truncated_residual_df["platform_community"]==platform+"_CVE"]
        coeff,pvalue = corrByRelationship(df1,df2)
        spearmann_df[platform+"_Crypto/"+platform+"_CVE_coef"]=spearmann_df["inf_relationship"].map(coeff)
        spearmann_df[platform+"_Crypto/"+platform+"_CVE_p"]=spearmann_df["inf_relationship"].map(pvalue)
    for community in ["Crypto","CVE"]:
        df1 = truncated_residual_df[truncated_residual_df["platform_community"]=="GitHub_"+community]
        df2 = truncated_residual_df[truncated_residual_df["platform_community"]=="Twitter_"+community]
        coeff,pvalue = corrByRelationship(df1,df2)
        spearmann_df["GitHub_"+community+"/Twitter_"+community+"_coef"]=spearmann_df["inf_relationship"].map(coeff)
        spearmann_df["GitHub_"+community+"/Twitter_"+community+"_p"]=spearmann_df["inf_relationship"].map(pvalue)
    spearmann_df=spearmann_df.fillna(0)
    spearmann_df["inf_relationship"]=spearmann_df["inf_relationship"].str.replace("creation","initiation") # replace the word "creation" by "initiation"
    spearmann_df.to_csv("spearmannStats.csv") 
    print("Spearmann's Correlation Test")
    print(spearmann_df)


    ##### Scatter plots of Median normalized total influence values across platforms and comminities by 
    #####influence relationships
    print("Plotting Meadian Normalized Total Influence Values Across Platforms and Communitites")
    for platform in ["GitHub","Twitter"]:
        df3= truncated_residual_df[truncated_residual_df["platform"]==platform].drop(columns="platform")
        df3=pd.melt(df3, id_vars=["level","community"], 
                value_vars=truncated_residual_df.columns[3:-1],var_name="inf_relationship", value_name="influence")
        df3 = pd.pivot_table(df3,values="influence", index=["level","inf_relationship"],columns="community").reset_index()
        df3["inf_relationship"]=df3["inf_relationship"].str.replace("creation","initiation")
        g = sns.FacetGrid(df3, col="inf_relationship",col_wrap=3)
        g = (g.map(plt.scatter, "CVE", "Crypto").set_titles("{col_name}"))
        print(platform)
        plt.show()    
    for community in ["Crypto", "CVE"]:
        df4= truncated_residual_df[truncated_residual_df["community"]==community].drop(columns="community")
        df4=pd.melt(df4, id_vars=["level","platform"], 
                value_vars=truncated_residual_df.columns[3:-1],var_name="inf_relationship", value_name="influence")
        df4 = pd.pivot_table(df4,values="influence", index=["level","inf_relationship"],columns="platform").reset_index()
        df4["inf_relationship"]=df4["inf_relationship"].str.replace("creation","initiation") 
        g = sns.FacetGrid(df4, col="inf_relationship",col_wrap=3)
        g = (g.map(plt.scatter, "GitHub", "Twitter").set_titles("{col_name}"))
        print(community)
        plt.show()


def Anova(empirical_inflcascades_df,scalefree_inflcascades_df):
    '''
    Three-way ANOVA: 
    '''
    scalefree_inflcascades_medians = (scalefree_inflcascades_df.groupby(by=["platform","community","level"])
                                      .median().fillna(0).reset_index())
    
    res_df = pd.DataFrame()
    for platform in ["GitHub","Twitter"]:
        for community in ["Crypto","CVE"]:
            df_SF = scalefree_inflcascades_medians[(scalefree_inflcascades_medians["platform"]==platform) & 
                                                   (scalefree_inflcascades_medians["community"]==community)]
            df_gt = empirical_inflcascades_df[(empirical_inflcascades_df["platform"]==platform) & 
                                              (empirical_inflcascades_df["community"]==community)]
            root_users=set(df_gt.root.unique())
            random.seed(0)
            random_root_users_sample = random.sample(root_users,k=40)
            for user in random_root_users_sample:
                user_df_gt = df_gt[df_gt["root"] == user]
                user_df_gt = user_df_gt.drop(columns="root")
                temp_residual_df = (user_df_gt.set_index(["platform","community","level"])
                .subtract(df_SF.set_index(["platform","community","level"]), fill_value=0).reset_index())
                res_df = res_df.append(temp_residual_df)
    res_df = res_df[res_df["level"]<=4]
    res_df["platform_community"] = res_df["platform"]+"_"+res_df["community"]
    anova_data=pd.melt(res_df, id_vars=["level","platform_community"], value_vars=relationshipTypes, var_name="inf_relationship", value_name="influence")

    ###### Normality check for each combination of groups of the platform_community, influence relationship and level ######
    threewayAnovaNormality=(anova_data.groupby(["platform_community","inf_relationship","level"])
             .apply(lambda x: pd.Series(scipy.stats.shapiro(x), index=['stats','P']))
             .reset_index())
    threewayAnovaNormality["H_0"] = threewayAnovaNormality["P"].apply(lambda x: "rejected" if x<0.05 else "not rejected")
    print("Three-way ANOVA Normality test", threewayAnovaNormality,sep='\n')
    print(f'{threewayAnovaNormality.groupby(["H_0"])["P"].count()}')
    threewayAnovaNormality = threewayAnovaNormality.rename(columns={"platform_community":"platformCommunity","inf_relationship":"infRelationship","H_0":"Null"})
    threewayAnovaNormality["platformCommunity"]=threewayAnovaNormality["platformCommunity"].str.replace("_","-")
    threewayAnovaNormality["stats"] = threewayAnovaNormality["stats"].map("{:.2f}".format)
    threewayAnovaNormality["P"] = threewayAnovaNormality["P"].map("{:.2E}".format)
    threewayAnovaNormality.to_csv("threewayANOVANormalityStats.csv",index=False)

    #### Normality check of residuals of the model####
    model = ols('influence~C(platform_community)+C(inf_relationship)+C(level)+C(platform_community):C(inf_relationship)+C(platform_community):C(level)+C(inf_relationship):C(level)+C(platform_community):C(inf_relationship):C(level)', data=anova_data).fit()
    print(f'Normality check of residuals of the three-way ANOVA model: {scipy.stats.shapiro(model.resid)}')
    fig = plt.figure(figsize= (10, 10))
    ax = fig.add_subplot(111)
    normality_plot, stat = scipy.stats.probplot(model.resid, plot= plt, rvalue= True)
    ax.set_title("Probability plot of model residual's", fontsize= 20)
    ax.set
    plt.show()