
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


class Preprocessor:

    def __init__(
            self,
            path: str,
            start_data=2,
            end_data=-4,
            sep=";"
    ):
        self.path = path
        self.start_data = start_data
        self.end_data = end_data
        self.sep = sep
        self.data = self.read_csv().loc[:, self.read_csv().columns != "Date"]

    def read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, sep=self.sep)
        return df[self.start_data:self.end_data]

    def prepare_for_daily_prediction(self):
        self.data = self.data["Open"]
        # result =
        count = 0

    def transform_data(self) -> pd.DataFrame:
        scaler = StandardScaler()
        # print(self.data)
        return pd.DataFrame(scaler.fit_transform(self.data.astype('float64')))

    def normalize_data(self, data):
        min = data.min()
        max = data.max()

        return (data - min) / (max - min)

    def min_max_scaler(self):
        minmax = MinMaxScaler(feature_range=(0, 1))
        self.data["scaled_open"] = minmax.fit_transform(self.data["Open"].astype('float64').values.reshape(-1, 1))

    def cluster_data_to_weeks(
            self,
            num_weeks=1,
            univariate=True
    ) -> pd.DataFrame:
        """
        loop over pandas and cluster data based on 5 days

        :param data:
        :param nums_weeks:
        :return:
        """
        result = {"open_values": [], "labels": []}
        count = 0
        if univariate:
            data = self.data["Open"]
        else:
            data = self.data

        while True:
            open_values = data[count: count + 5 * num_weeks]

            if not univariate:
                values = []
                for i in range(5 * num_weeks):
                    values.append(open_values.iloc[i].to_list())

                open_values = values
            else:
                open_values = open_values.to_list()

            if len(open_values) < 5 * num_weeks:
                break

            labels = data[count + 5 * num_weeks: count + 5 * (num_weeks + 1)]
            if not univariate:
                labels = labels["Open"]

            if len(labels) < 5:
                break

            result["open_values"].append(open_values)
            result["labels"].append(labels.to_list())
            count += 5

            try:
                data[count]
            except KeyError:
                break

        # print(result)

        return pd.DataFrame(result)

    def cluster_data_to_train_test(self, num_weeks=1, univariate=True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        data = self.cluster_data_to_weeks(num_weeks, univariate)
        train_df = data.sample(frac=0.8)
        test_df = data.drop(train_df.index)
        # X_train, X_test = train_test_split(d, test_size=0.2)
        X_train = torch.tensor(train_df["open_values"].to_list()).to(device)
        y_train = torch.tensor(train_df["labels"].to_list()).to(device)
        X_test = torch.tensor(test_df["open_values"].to_list()).to(device)
        y_test = torch.tensor(test_df["labels"].to_list()).to(device)
        return X_train, y_train, X_test, y_test
