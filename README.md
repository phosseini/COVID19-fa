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
* `tags.txt`: list of Twitter hashtags in Persian and English on Iran COVID-19. 

#### List of `tweet id` files
| File name | # of tweet ids | Time period |
| :---:         |     :---:      |          :---: |
| `../data/tweet_ids_v1.0.txt` | 530,249 | March 13, 2020 - April 19, 2020 |
| `../data/tweet_ids_v2.0.txt` | 1,441,426 | March 13, 2020 - November 15, 2020 |
| `../data/tweet_ids_v3_1.txt` | 952,333 | March 13, 2020 - April 22, 2021 (Part 1) |
| `../data/tweet_ids_v3_2.txt` | 952,335 | March 13, 2020 - April 22, 2021 (Part 2) |
| `../data/tweet_ids_v4.0.txt` | 286,645 | April 23, 2021 - January 3, 2022 <sup>:warning:</sup> |

:warning:&nbsp;There are some missing tweets from early 2021-06 to late 2021-08 since our tweet collection servers were temporarily down.

### Requirements for LDA analysis
* We use Gensim's python wrapper for `Mallet` in our topic modeling. Please make sure you have properly installed Mallet before running LDA topic modeling examples. You can find more instructions [here](https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html) and the official guide to installing Mallet [here](http://mallet.cs.umass.edu/download.php).

### How to cite our work?
You can cite our [paper](https://www.aclweb.org/anthology/2020.nlpcovid19-2.26/):

```bibtex
@inproceedings{hosseini-etal-2020-content,
    title = "Content analysis of {P}ersian/{F}arsi Tweets during {COVID}-19 pandemic in {I}ran using {NLP}",
    author = "Hosseini, Pedram and Hosseini, Poorya and Broniatowski, David",
    booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID}-19 (Part 2) at {EMNLP} 2020",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlpcovid19-2.26",
}
```
