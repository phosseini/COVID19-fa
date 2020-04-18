# Iran and Coronavirus (COVID-19) on Social Media 

As COVID-19 is spreading globally, this now pandemic is shaking up all aspects of daily life in affected countries. While each country will go through its own unique experience, there may be much shared on how different populations react to this pandemic. Iran, along with China, South Korea, and Italy has been among the countries that have been hit hardest in the early wave of COVID-19 spread. Leveraging Machine Learning and Natural Language Processing techniques, we are conducting an ongoing analysis of the reaction of the Persian/Farsi speaking users on social media starting with the case of Twitter.

## Analysis and Reproducibility reports
### List of jupyter notebooks:
* `statistics.ipynb`: visualizing Iran related COVID-19 statistics including the number of infected individuals, death, and recovered cases. Also, this notebook includes the visualization of the number of posted `original` tweets over time.
* `lda_analysis.ipynb`: topic modeling using Latent Dirichlet Allocation (LDA). An interactive html file of all topics is also available at `lda.html`.
* `clustering.ipynb`: clustering tweets using K-means/MiniBatchKMeans
* `content_analysis.ipynb`: analysis of the content of tweets
* `hashtags.ipynb`: listing and processing COVID-19 related Persian hashtags on Twitter
* `pre_processing.ipynb`: testing different ways of pre-processing Persian tweets

### Tweet IDs
* `data/tweet_ids_v1.txt`: ids of the tweets we used in the first round of analysis.
