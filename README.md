# The-Oracle 
---

# Sentiment Analysis on Reddit Posts with BERT

## Introduction

In the vast realm of the stock market, where numbers and charts reign supreme, we embarked on a quest to understand the power of sentiment. Can the collective mood of investors, as captured on platforms like Reddit, influence the stock market? This project dives deep into this question, leveraging the prowess of BERT and the raw, unfiltered emotions of Reddit.

## Motivation

The stock market, while seemingly a world of numbers, is driven by human emotions - fear, greed, euphoria, and panic. Traditional financial news outlets provide polished, edited narratives. Reddit, however, offers a raw, direct insight into the psyche of the investor. By analyzing this sentiment, we aimed to uncover patterns and relationships that might hint at the market's next move.

## Approach

We adopted an 8-class sentiment classification approach, encompassing:

- Bullish
- Bearish
- Neutral
- Speculative
- Stable
- Uncertain
- Overbought
- Oversold

## Key Highlights

### Data Collection

We sourced our data from two popular subreddits: `stocks` and `wallstreetbets`. Using the `praw` library, we fetched the latest posts:

```python
reddit = praw.Reddit(client_id="...", client_secret="...", user_agent="...")
subreddits = ['stocks', 'wallstreetbets']
titles, dates = [], []
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.new(limit=300):
        titles.append(post.title)
        dates.append(datetime.fromtimestamp(post.created_utc).date())
```

### Sentiment Analysis with OpenAI

To classify the sentiment of each post, we leveraged OpenAI's GPT-3:

```python
openai.api_key = '...'
def analyze_sentiment(title):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""Classify the sentiment of this stock market related text: \"{title}\"...""",
        max_tokens=50
    )
    return response.choices[0].text.strip()
```

### Data Visualization

We visualized the distribution of sentiments using `seaborn`:

```python
sns.countplot(data=RedditData, x='sentiment_class', order=order_categories, palette="deep")
```

### BERT vs. Logistic Regression

BERT's deep understanding of context allowed it to outshine Logistic Regression. We fine-tuned BERT on our dataset:

```python
tokenizer = BertTokenizer.from_pretrained("./bert_finetuned")
model = BertForSequenceClassification.from_pretrained('./bert_finetuned', num_labels=len(label_map))
```

### Hyperparameter Tuning with Optuna

To optimize our model's performance, we employed Optuna for hyperparameter tuning:

```python
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    ...
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
    return trainer.evaluate()["eval_loss"]
```

### Market Analysis

We merged our sentiment data with the S&P 500's historical data to analyze potential correlations:

```python
merged = pd.merge(sp_data, daily_sentiment, left_on='Date', right_on='date')
```

## Conclusion

Our exploration unveiled a subtle relationship between Reddit sentiment and the S&P 500's movements. While the market doesn't dance to Reddit's every tune, there's a discernible influence. This project is a mere scratch on the surface, a starting point for deeper dives and refined explorations. The stock market is a complex entity, and while numbers play a significant role, the human element - sentiment - cannot be ignored.

