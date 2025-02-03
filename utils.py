import pandas as pd
import numpy as np

class TradingIndicators:
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On Balance Volume.

        :param data: DataFrame with 'Close' and 'Volume' columns.
        :return: OBV as a pandas series.
        """
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=data.index)
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        :param data: DataFrame with 'High', 'Low', and 'Close' columns.
        :return: ADX values as a pandas series.
        """
        high = data['High']
        low = data['Low']
        close = data['Close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        pos_dm = high.diff()
        neg_dm = low.diff()
        pos_dm[pos_dm <= 0] = 0
        neg_dm[neg_dm >= 0] = 0
        neg_dm = abs(neg_dm)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        dx = (abs(pos_dm - neg_dm) / (pos_dm + neg_dm)) * 100

        adx = dx.rolling(window=period).mean()
        
        return adx
