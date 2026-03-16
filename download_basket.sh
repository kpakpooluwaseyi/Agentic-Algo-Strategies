#!/bin/bash

# Crypto
./venv/bin/python download_data.py BTC-USD 15m --period 59d
./venv/bin/python download_data.py ETH-USD 15m --period 59d
./venv/bin/python download_data.py SOL-USD 15m --period 59d
./venv/bin/python download_data.py DOGE-USD 15m --period 59d
./venv/bin/python download_data.py PEPE-USD 15m --period 59d

# Tech / Growth Stocks
./venv/bin/python download_data.py NVDA 15m --period 59d
./venv/bin/python download_data.py TSLA 15m --period 59d
./venv/bin/python download_data.py AAPL 15m --period 59d

# Meme / Volatility (Low Cap Proxy)
./venv/bin/python download_data.py GME 15m --period 59d
# ./venv/bin/python download_data.py AMC 15m --period 59d

# ETFs / Indices
./venv/bin/python download_data.py SPY 15m --period 59d
./venv/bin/python download_data.py QQQ 15m --period 59d
./venv/bin/python download_data.py TQQQ 15m --period 59d
./venv/bin/python download_data.py IWM 15m --period 59d
./venv/bin/python download_data.py TLT 15m --period 59d

# Commodities / Metals (ETFs used for reliability)
./venv/bin/python download_data.py GLD 15m --period 59d
./venv/bin/python download_data.py SLV 15m --period 59d
./venv/bin/python download_data.py USO 15m --period 59d

# Forex
./venv/bin/python download_data.py EURUSD=X 15m --period 59d
