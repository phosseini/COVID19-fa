# Iran and COVID-19 on Social Media 

As COVID-19 is spreading globally, this now pandemic is shaking up all aspects of daily life in affected countries. While each country will go through its own unique experience, there may be much shared on how different populations react to this pandemic. Iran, along with China, South Korea, and Italy has been among the countries that have been hit hardest in the early wave of COVID-19 spread. Leveraging Machine Learning and Natural Language Processing techniques, we are conducting an ongoing analysis of the reaction of the Persian/Farsi speaking users on social media starting with the case of Twitter.


## Analysis and Reproducibility reports
### List of jupyter notebooks in `notebooks` folder:
* `statistics.ipynb`: visualizing Iran related COVID-19 statistics including the number of infected individuals, death, and recovered cases. Also, this notebook includes the visualization of the number of original tweets, retweets, reply, and quotes in Persian on COVID-19 over time.
* `lda_analysis.ipynb`: topic modeling using Latent Dirichlet Allocation (LDA).
* `clustering.ipynb`: clustering tweets using K-means/MiniBatchKMeans.
* `content_analysis.ipynb`: analysis of the content of tweets.
* `hashtags.ipynb`: listing and processing COVID-19 related Persian hashtags on Twitter.
* `pre_processing.ipynb`: testing different ways of pre-processing Persian tweets.

### List of files in `data` folder:
* `persian_stop_words.txt`: list of Persian stop words.
* `tags.txt`: list of Twitter hashtags in Persian and English on Iran CVOID-19. 
* `tweet_ids_v1.0.txt`: ids of the tweets we used in the first round of analysis.

### How to cite our work?
You can cite our [arXiv paper](https://arxiv.org/abs/2005.08400):

```
@article{hosseini2020content,
  title={Content analysis of Persian/Farsi Tweets during COVID-19 pandemic in Iran using NLP},
  author={Hosseini, Pedram and Hosseini, Poorya and Broniatowski, David A},
  journal={arXiv preprint arXiv:2005.08400},
  year={2020}
}
```
