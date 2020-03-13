import string
import re


def remove_url(text):
    text = ' '.join(x for x in text.split() if x.startswith('http') == False and x.startswith('www') == False)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # extra step to make sure html tags are completely removed
    clean = re.compile('<.*>|<.*\"')
    result = re.sub(clean, '', text)
    return result


def clean_persian_tweets(tweet):
    # removing URLs
    tweet = remove_url(tweet)

    # removing non-relevant punctuation marks
    puncs = list("؟!,،?.؛")
    for punc in puncs:
        tweet = tweet.replace(punc, " ")

    # removing more than one space
    tweet = ' '.join(tweet.split())

    text_tokens = tweet.split()

    cleaned_text = []
    for token in text_tokens:
        if not token.startswith("@") and not token.startswith("https") and not token.startswith(
                "&amp") and token != "RT":
            cleaned_text.append(token)

    cleaned_tweet = re.sub(' +', ' ', ' '.join(cleaned_text))
    return cleaned_tweet
