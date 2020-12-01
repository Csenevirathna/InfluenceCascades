# Influence Cascades
This code is implemented to explore the cascading effect of soical influence in online social networks. We use initiation, contribution and sharing action classification to define influence relationships and input_data folder contains the magnitude of influence for each of these influence relationships between users in the interest communities and platforms. Influence_Cascade_Extraction_and_Visualization.ipynb extract Influence cascades from these networks, visualize the user distributions, influence distributions and, perform statistical tests for further analysis.  
## Required python packages
The following python packages are required to run the Influence_Cascade_Extraction_and_Visualization.ipynb.
* statsmodels==0.12.1 
* pandas
* plotly==3.6.0
* networkx
* numpy
* seaborn
* scipy

Dependencies for statsmodels are python >=3.6, Numpy >=1.15, SciPy >=1.2, Pandas >=0.23, Patsy >=0.5.1

## Other requirements
Visualization of Influence Cascades use plotly chart-studio and it is needed plotly account.    
To create free plotly account: https://chart-studio.plotly.com/Auth/login/?action=signup&next=%2Fsettings%2Fapi#/

## Running Influence_Cascade_Extraction_and_Visualization.ipynb
* For the first time use of plotly chart-studio: uncomment the line "#plotly.tools.set_credentials_file(username='username', api_key='api_key')" inside the "plotSankey" function and set your credentials.
* Execute all the cells
