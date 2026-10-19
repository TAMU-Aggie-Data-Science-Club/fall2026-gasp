import io
import requests
import pandas as pd

def parse_cpc_daily(text: str) -> pd.DataFrame:
    #creates variable lines to hold list of the items in the file
    lines = text.strip().splitlines()

    #intializing variables 
    raw_measure = None
    raw_weighting = None
    header_end_idx = 0
    #for loop looks for the header file and adds what type of data is being store
    for idx, line in enumerate(lines):
        #check for the product, adding only the title
        if line.startswith("Product:"):
            raw_measure = line.split(":", 1)[1].strip()
        #check for the weighted, adding only the title
        elif line.startswith("Weights:"):
            raw_weighting = line.split(":", 1)[1].strip()
        #once the code hits dates, it has reached the beginning of the actual data table
        elif line.lstrip().startswith(("YYYYMMDD", "Date" , "Region")):
            header_end_idx = idx
    #everything from the header on is connected into one text
    data_text = "\n".join(lines[header_end_idx:])
    #the text is fed into pandas as a file and parsed using | as the dividers
    df = pd.read_csv(io.StringIO(data_text), sep="|")
    #clean up any remaining whitespace that may cause issues later 
    df.columns = df.columns.str.strip()
    # grabs the first colum listed, should be the date column
    date_col = df.columns[0]
    #if any date column is missing or blank it is deleted to not intefere with the code
    df = df.dropna(subset=[date_col])

    #grab the name of the first column
    id_col = df.columns[0]
    #creates a list of the columns 
    date_cols = [c for c in df.columns if c != id_col]
    #pandas melt function to convert wide to long 
    df_long = df.melt(
        id_vars=[id_col],
        value_vars=date_cols,
        var_name="date",
        value_name="value"
    )
    
    #renaming columns 
    df_long = df_long.rename(columns={id_col: "region"})
    df_long["measure"] = raw_measure
    df_long["weighting"] = raw_weighting
    
    #convert date to actual datetime objects
    df_long["date"] = pd.to_datetime(df_long["date"], format="%Y%m%d")
    
    #change the values to type float
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    
    #reorder the list and sort by the oldest
    df_long = df_long[["date", "region", "measure", "weighting", "value"]]
    df_long = df_long.sort_values("date").reset_index(drop=True)
    
    return df_long


def fetch_cpc_daily(file: str, start_year: int, end_year: int) -> pd.DataFrame:
    pass