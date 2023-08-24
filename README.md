# 📺 ChatJerryPT

This repo is an implementation of a locally hosted Jerry Seinfeld chatbot.

![Jerry working on a computer](https://i.stack.imgur.com/bkS0K.gif)

## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
1. Run the app: `make start`
   1. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`. You can find more documentation [here](https://langchain.readthedocs.io/en/latest/tracing.html).
1. Open [localhost:9000](http://localhost:9000) in your browser.
