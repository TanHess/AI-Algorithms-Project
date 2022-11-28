import pandas as pd


def replace_col(df, replacing,to_replace,if_present=False):

    if if_present == True:
        for index,row in df.iterrows():
            if pd.isna(row[replacing]) == False:
                df.loc[index,to_replace] =  row[replacing]
        return df
    else:
        for index,row in df.iterrows():
            df.loc[index,to_replace] = row[replacing]
        return df
    


def main():
    file_name = input("Enter your file name: ")
    new_file_name = file_name.split(sep=".")[0] + '_new.csv'

    df = pd.read_csv(file_name)
    to_replace = input("Enter the column you wish to replace: ")
    replacing = input("Enter the column you wish to replace the previous column: ")
    df = replace_col(df, replacing, to_replace, if_present=True)

    df.to_csv(new_file_name, index=False)
    print("Done writing to:",new_file_name)
    
if __name__ == '__main__':
    main()