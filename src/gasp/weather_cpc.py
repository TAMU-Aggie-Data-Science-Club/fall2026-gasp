import io
import requests
import pandas as pd
import datetime

def parse_cpc_daily(text: str) -> pd.DataFrame:
    lines = text.strip().splitlines()
    #intialize variables
    raw_measure = "unknown"
    raw_weighting = "unknown"
    header_end_idx = 0
    
    # extract headers and locate where the region begins
    for idx, line in enumerate(lines):
        if line.startswith("Product:"):
            val = line.split(":", 1)[1].strip().lower()
            raw_measure = "hdd" if "heating" in val else "cdd" if "cooling" in val else val
        elif line.startswith("Weights:"):
            raw_weighting = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Region|") or line.lstrip().startswith("Region|"):
            header_end_idx = idx
            break
            
    # if region wasn't caught correctly 
    if header_end_idx == 0:
        pipe_counts = [line.count('|') for line in lines]
        if max(pipe_counts) > 0:
            header_end_idx = max(range(len(pipe_counts)), key=pipe_counts.__getitem__)
            
    # clean up the table by skipping empty and dashed lines and stripping trailing pipes
    cleaned_lines = []
    for line in lines[header_end_idx:]:
        stripped = line.strip()
        if not stripped or all(c in '-' for c in stripped):
            continue
        if stripped.endswith('|'):
            stripped = stripped[:-1]
        cleaned_lines.append(stripped)
        
    data_text = "\n".join(cleaned_lines)
    
    # parse with pandas 
    df = pd.read_csv(
        io.StringIO(data_text), 
        sep="|", 
        engine="python", 
        skipinitialspace=True, 
        on_bad_lines="skip"
    )
    
    # clean the white space in the columns and drop unamed 
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Melt wide date columns to long format
    id_col = df.columns[0]
    df = df.dropna(subset=[id_col])
    date_cols = [c for c in df.columns if c != id_col]
    
    df_long = df.melt(
        id_vars=[id_col],
        value_vars=date_cols,
        var_name="date",
        value_name="value"
    )
    
    # make the table fit the format
    df_long = df_long.rename(columns={id_col: "region"})
    df_long["measure"] = raw_measure
    df_long["weighting"] = raw_weighting
    
    # change conus to us
    df_long["region"] = df_long["region"].astype(str).str.strip()
    df_long["region"] = df_long["region"].apply(lambda r: "US" if r.upper() == "CONUS" else r)
    
    # filter invalid rows and convert to type datetiem and float
    df_long["date"] = pd.to_datetime(df_long["date"], format="%Y%m%d", errors="coerce")
    df_long = df_long.dropna(subset=["date"])
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    
    # reorder and sort oldest to most recent
    df_long = df_long[["date", "region", "measure", "weighting", "value"]]
    df_long = df_long.sort_values("date").reset_index(drop=True)
    
    return df_long

def fetch_cpc_daily(file: str, start_year: int, end_year: int) -> pd.DataFrame:
    all_dfs = []
    current_year = datetime.date.today().year
    
    for year in range(start_year, end_year + 1):
        # loops through from the start year to end year getting the requested files 
        url = f"https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{year}/{file}"
        #checks if grabbing the data was successful 
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch data for year {year} from {url}: {e}")
            
        # parse using the parse function
        df_year = parse_cpc_daily(response.text)
        
        # checks if we are in theb current year, then filters out all the days that haven't happened
        if year == current_year:
            today = pd.Timestamp.today().normalize()
            df_year = df_year[df_year["date"] <= today]
            df_year = df_year.dropna(subset=["value"])
            
        all_dfs.append(df_year)
        
    if not all_dfs:
        return pd.DataFrame(columns=["date", "region", "measure", "weighting", "value"])
        
    # combine all the data into one dataframe
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # removes duplicates and sorts from oldest to most recent
    combined_df = combined_df.drop_duplicates(subset=["date", "region"]).sort_values("date").reset_index(drop=True)
    
    return combined_df