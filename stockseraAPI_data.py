import stocksera

client = stocksera.Client(api_key="aKyxZL4D.XZpFadsSHASGAokKYWXSnyDDLih0swAk")

# data = client.wsb_mentions(days=1, ticker="AMZN")


data = client.subreddit(days=1500, ticker="AMZN")
print(len(data))

for d in data:
    print(d)
