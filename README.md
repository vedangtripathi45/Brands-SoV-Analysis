## Brands-SoV-Analysis
Analysis of metrics and share of voice calculation for fan manufacturing brands over YouTube Searches from different keywords.

# Included Files
**AI_AGENT_SoV.py**: Main python file.
**Teminal Output.txt**: Contains the output that appears on the terminal window upon running the program.
**sov_analysis_results.png**: Shows the evaluated data in visualized format through bar graphs.
**sov_detailed_results.json**: Contains the output results in a detailed manner to show that which isn't shown in the png file.

#  Installation of required dependencies in Windows
pip install google-api-python-client pandas matplotlib seaborn transformers torch

# Brands Searched
Atomberg, Havells, Crompton, Usha, Orient, Bajaj

# Key Points
This program analyzes different metrics in a video, combines them to calculate SoV for each brand. Sentiment analysis is done for each brand in the video's content and comments to get positive/negative share of the brands. The brands, keywords, number of search results and comments to be analyzed can be easily changed. 
