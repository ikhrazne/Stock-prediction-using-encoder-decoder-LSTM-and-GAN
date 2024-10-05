
import yfinance as yf


def get_stock_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    return data


def save_data(data, path):
    data.to_csv(path)


if __name__ == "__main__":

    data = get_stock_data("AAPL", "2018-01-02")
    save_data(data, "AAPL.csv")
    print(data.head())
