import pandas as pd

def get_data():
    reader = pd.read_csv("data/warfarin.csv")
    print(reader)




if __name__ == "__main__":

    get_data()














