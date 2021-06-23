import pandas as pd


def dataset_unification(dataframe1, dataframe2):
    new_file_name = "datasetPostUnified.xlsx"

    data_unified = pd.merge(dataframe1, dataframe2, on="Permalink")
    data_unified.to_excel("datasetPostUnified.xlsx", index=False)

    data_unified.to_excel(new_file_name, index=False, encoding="utf-8")


if __name__ == '__main__':
    data_general = pd.read_excel("datasetPostGeneral.xlsx", engine="openpyxl")
    data_complemento = pd.read_excel("datasetPostComplemento.xlsx", engine="openpyxl")

    dataset_unification(data_general, data_complemento)
