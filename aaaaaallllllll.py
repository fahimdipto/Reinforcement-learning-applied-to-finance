import alpaca_trade_api as tradeapi
import time

if __name__ == '__main__':
    """
    With the Alpaca API, you can check on your daily profit or loss by
    comparing your current balance to yesterday's balance.
    """

    # First, open the API connection
    api = tradeapi.REST(
        'PKFJFVQVL9EXXFAERYT2',
        '/8OhhH1fUnlrbsL4JCQwAu14fQOvj3aECfM5R2RC',
        'https://paper-api.alpaca.markets'
    )

    # The security we'll be shorting
    symbol = 'AAPL'

    # # Submit a market order to open a short position of one share
    # order = api.submit_order(symbol, 1, 'sell', 'market', 'day')
    # print("Market order submitted.")
    #
    # # Submit a limit order to attempt to grow our short position
    # # First, get an up-to-date price for our symbol
    # symbol_bars = api.get_barset(symbol, 'minute', 1).df.iloc[0]
    # symbol_price = symbol_bars[symbol]['close']
    # # Submit an order for one share at that price
    # order = api.submit_order(symbol, 1, 'sell', 'limit', 'day', symbol_price)
    # print("Limit order submitted.")
    #
    # # Wait a second for our orders to fill...
    # print('Waiting...')
    # time.sleep(1)

    # Check on our position
    position = api.get_position(symbol)
    if int(position.qty) < 0:
        print(f'Short position open for {symbol}')



import alpaca_backtrader_api
import backtrader as bt
from datetime import datetime

ALPACA_API_KEY = 'PKFJFVQVL9EXXFAERYT2'
ALPACA_SECRET_KEY = '8OhhH1fUnlrbsL4JCQwAu14fQOvj3aECfM5R2RC'
ALPACA_PAPER = True


class SmaCross(bt.SignalStrategy):
  def __init__(self):
    sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
    crossover = bt.ind.CrossOver(sma1, sma2)
    self.signal_add(bt.SIGNAL_LONG, crossover)


cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

store = alpaca_backtrader_api.AlpacaStore(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=ALPACA_PAPER
)

if not ALPACA_PAPER:
  broker = store.getbroker()  # or just alpaca_backtrader_api.AlpacaBroker()
  cerebro.setbroker(broker)

DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
data0 = DataFactory(dataname='AAPL', historical=True, fromdate=datetime(
    2015, 1, 1), timeframe=bt.TimeFrame.Days)
cerebro.adddata(data0)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.plot()


