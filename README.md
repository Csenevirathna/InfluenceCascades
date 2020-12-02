# Influence Cascades
This code is implemented to explore the cascading effect of soical influence caused by actions in online social networks. We use initiation, contribution and sharing action classification to define influence relationships.

## Required python packages
The following python packages and versions are required to run the Influence_Cascade_Extraction_and_Visualization.ipynb.
* statsmodels==0.12.1 
* plotly==3.6.0
* pandas
* networkx
* numpy
* seaborn
* scipy

Dependencies for statsmodels are python >=3.6, Numpy >=1.15, SciPy >=1.2, Pandas >=0.23, Patsy >=0.5.1

## Other requirements
To visualize the Influence Cascades, plotly chart-studio (online) is used and it is needed a plotly account.    
To create a free account: https://chart-studio.plotly.com/Auth/login/?action=signup&next=%2Fsettings%2Fapi#/

* To find account credentials: sign in to the account->go to settings->go to API keys
* The first time users of plotly chart-studio (online), set your credentials as follows: 
    * Go to infcascade.py in "module" folder
    * Go to "plotSankey" function
    * Uncomment the line "#plotly.tools.set_credentials_file(username='username', api_key='api_key')" and replace 'username' and 'api_key' with your credentials.

## Data
"input_data" folder contains the magnitude of influence for each of the influence relationships between users in the interest communities and platforms. 

## Modules
"module" folder contains the modules which use in Influence_Cascade_Extraction_and_Visualization.ipynb
* initialization.py: the set of functions to extract influence cascades and to create dataframes for analysis
* analyze.py: the set of functions for statistical testing 
* infcascades.py: the set of functions to visualize influence cascades
* gloabvars.py: contains global variables

## Running Influence_Cascade_Extraction_and_Visualization.ipynb
This will extract Influence cascades from the interest networks, visualize the user distributions, influence distributions and, perform statistical tests for further analysis.
1. To install all the required python packages, uncomment the cell which, contains the command lines to install packges (press ctrl+/). 
2. Execute all the cells
    * Set your plotly chart studio credentials before the execution, if necessary (Refer: other requirments).
