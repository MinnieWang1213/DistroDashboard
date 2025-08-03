import streamlit as st
import os
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from probability_matrix import GetMatrix
import custom_filtering_dataframe
from returns_main import folder_input,folder_processed_pq
import requests
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import percentileofscore,skew,kurtosis


st.cache_data.clear()

# Defining custom functions to modify generated data as per user input
def get_volatility_returns_csv_stats_custom_days(target_csv,target_column):
        
    stats_csv=target_csv[target_column].describe(percentiles=[0.1,0.25,0.5,0.75,0.95,0.99])
    # Add additional statistics to the DataFrame
    stats_csv.loc['mean'] = target_csv[target_column].mean()
    stats_csv.loc['skewness'] = target_csv[target_column].skew()
    stats_csv.loc['kurtosis'] = target_csv[target_column].kurtosis()

    stats_csv.index.name = 'Volatility of Returns Statistic'
    return stats_csv

def get_volatility_returns_csv_custom_days(target_csv,target_column):
    target_csv['ZScore wrt Given Days']=(target_csv[target_column]-target_csv[target_column].mean())/target_csv[target_column].std()
    return target_csv

# Defining functions to download the data

# 1. Function to convert DataFrame to Excel file with multiple sheets
# def download_combined_excel(df_list,sheet_names,skip_index_sheet=[]):
#     # Create a BytesIO object to hold the Excel file
#     output = BytesIO()

#     # Create a Pandas Excel writer using openpyxl as the engine
#     with pd.ExcelWriter(output, engine='openpyxl') as writer:
#         for sheetname,mydf in zip(sheet_names,df_list):
#             if sheetname in skip_index_sheet:
#                 mydf.to_excel(writer, sheet_name=sheetname,index=False)
#             else:
#                 mydf.to_excel(writer, sheet_name=sheetname)
#     # Save the Excel file to the BytesIO object
#     output.seek(0)
#     return output

def download_combined_excel(df_list, sheet_names, skip_index_sheet=[]):
    output = BytesIO()

    # Use xlsxwriter for styling
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        for sheetname, df in zip(sheet_names, df_list):

            df.to_excel(writer, sheet_name=sheetname, index=(sheetname not in skip_index_sheet))
            worksheet = writer.sheets[sheetname]

            # Apply bold to the last row (only for 'Volatility Returns')
            if sheetname == 'Volatility Returns':
                bold_format = workbook.add_format({'bold': True})
                last_excel_row = len(df)
                worksheet.set_row(last_excel_row, None, bold_format)

    output.seek(0)
    return output

# 2.1 Main function to read image url and download as png files
def process_images(image_url_list):
    # Logic for downloading image bytes
    st.session_state["image_bytes_list"] = get_image_bytes(image_url_list)
    st.session_state["button_clicked"] = False  # Reset the button state after processing is complete

# 2.2 Function to get image bytes from list of images.
def get_image_bytes(image_url_list):
    image_bytes = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_image, image_url_list)
        for result in results:
            if result:
                image_bytes.append(result)
    return image_bytes

# 2.3 Function to fetch image url
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)  # Add a timeout to prevent hanging
        response.raise_for_status()  # Raise HTTP errors if any
        image = Image.open(BytesIO(response.content))  # Open the image
        output = BytesIO()
        image.save(output, format='PNG')  # Save the image in PNG format
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error processing image {url}: {e}")
        return None
    
# 2.4 Function to download image created via matplotlib.
def download_img_via_matplotlib(plt_object):
    buf=BytesIO()
    plt_object.savefig(buf, format="png",bbox_inches='tight')
    buf.seek(0)  # Go to the beginning of the buffer
    return buf

#2.5 used to add the sessions column in case all data is considered.
def get_session(timestamp):
    hour = timestamp.hour
    # minute = timestamp.minute
    if 18 <= hour < 24:
        return "Asia 18-24 ET"
    elif 0 <= hour < 7:  # or (hour == 6 and minute < 30):
        return "London 0-7 ET"
    elif 7 <= hour < 10:  # or (hour == 6 and minute >= 30):
        return "US Open 7-10 ET"
    elif 10 <= hour < 15:
        return "US Mid 10-15 ET"
    elif 15 <= hour < 17:
        return "US Close 15-17 ET"
    else:
        return "Other"


# 3. Function to create a ZIP file (not used)
def create_zip(excel_file_list, image_bytes_list):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        # Add Excel file
        for excel_file in excel_file_list:
            zip_file.writestr('combined_data.xlsx', excel_file.getvalue())
        # Add image file
        for image_bytes in image_bytes_list:
            zip_file.writestr('example_image.png', image_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

#5.1.1 helper function for 5.1
def add_start_end_ts(all_event_ts , delta):

    if(delta < 0):  # pre event + custom with delta < 0

        all_event_ts['end'] = all_event_ts['datetime'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0)) 
        all_event_ts['start'] = all_event_ts['end'] + pd.Timedelta(hours = delta)

    else:   # immediate reaction + custom with delta > 0

        all_event_ts['start'] = all_event_ts['datetime'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0))
        all_event_ts['end'] = all_event_ts['start'] + pd.Timedelta(hours = delta)

    return all_event_ts

#5.1.2 helper function for 5.1
def isolate_event(event_ts , time_gap_hours , selected_event , sub_event_dic):
    
    df = event_ts.copy()
    event_list = list(sub_event_dic.keys()) # list of events available in the drop down.
    event_list_lower = [e.strip().lower() for e in event_list if e.strip().lower() != selected_event.lower()]

    clean_rows = []
    counter1 = 0
    counter2 = 0
    for _, row in df.iterrows():

        cleaned_event_name = row['events'].replace(" ", "").strip().lower()[:-3]
        include = True
        
        t = row['datetime']
        t_minus = t - pd.Timedelta(hours=time_gap_hours)
        t_plus = t + pd.Timedelta(hours=time_gap_hours)

        if any(cleaned_event_name == s.strip().lower().replace(" ", "") for s in sub_event_dic[selected_event]):
            # Get full window, including current row
            nearby_events = df[
                (df['datetime'] >= t_minus) &
                (df['datetime'] <= t_plus)
            ]

            # Check if any unwanted event appears in the entire window
            include = not (
                nearby_events['events']
                .str.strip().lower()
                .apply(lambda x: any(e in x for e in event_list_lower))
                .any()
            )
        else:
            continue

        if include:
            clean_rows.append(row)
            counter1 += 1
        else:
            counter2 += 1

    # Keep only rows with no unwanted overlap
    filtered_df = pd.DataFrame(clean_rows)

    return filtered_df

#5.1 calculating the returns for event specific distros
def calc_event_spec_returns(selected_event , all_event_ts , ohcl_1h , delta = 1, filter_out_other_events=False,  time_gap_hours=2 , sub_event_filtering_dict = {} , sub_event_dict = {}):
    """
    Computes event-specific returns from OHLC data based on economic event filters and sub-event conditions.

    This function:
    - Filters and isolates economic events based on the main event and corresponding sub-event data.
    - Applies time windows (pre, during, or custom) around events.
    - Filters events based on whether each selected sub-event meets a specified condition (how much otter/cooler than expected).
    - Computes return metrics (volatility return, absolute return, and signed return) for each valid event window using OHLC data.

    Parameters:
    ----------
    selected_event : str
        The main economic event selected (e.g., "CPI", "NFP").
    
    all_event_ts : pd.DataFrame
        DataFrame of all economic events with at least the following columns:
        ['datetime', 'events', 'actual', 'consensus' or 'forecast'].

    ohcl_1h : pd.DataFrame
        Hourly OHLC data with a 'US/Eastern Timezone' timestamp column and columns:
        ['Open', 'High', 'Low', 'Close'].

    mode : int
        Controls how start and end timestamps are calculated:
        - 1: Pre-event mode (start = event - 8 hours)
        - 2: During-event mode (start = event + 1 hour)
        - Else: Custom mode using `delta` value.

    delta : int, optional (default=0)
        Number of hours to shift for custom event windows (only used if `mode` not 1 or 2).

    filter_out_other_events : bool, optional (default=False)
        Whether to filter out overlapping or closely timed events.

    time_gap_hours : int, optional (default=2)
        Event instance removed if there is another event in this window around the event being considered (used if `filter_out_other_events=True`).

    sub_event_filtering_dict : dict
        Dictionary mapping each selected sub-event to a 2-element list `[lower_bound, upper_bound]` for filtering based on:
        `actual - consensus` (or `actual - forecast` if `consensus` is NaN) deviation.

    sub_event_dict: dict
        Dictionary with the events as keys and associated sub-events list as value. Used for event isolation.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with calculated return metrics for each event:
        - 'Volatility Return'
        - 'Absolute Return'
        - 'Return'
        - 'Start_Date'
        - 'End_Date'
        - 'Entry_Price'
        - 'Exit_Price'
        - 'High'
        - 'Low'

        Returns an empty DataFrame if no events satisfy the sub-event conditions.
    """

    event_ts = all_event_ts.copy()

    event_ts.events = event_ts.events.astype(str)

    print("Event_ts columns: " , event_ts.columns)

    event_ts = event_ts.dropna(subset=['events'])

    event_ts = event_ts.drop_duplicates(subset=['datetime','events'], keep='last')

    cutoff_time = pd.to_datetime('2022-12-20 00:00:00-05:00', errors='coerce')
    event_ts = event_ts[event_ts['datetime'] >= cutoff_time]

    ## DEFINING THE START & END TIMESTAMPS FOR ANALYSIS:

    #pre event
    # if(mode == 1):
    #     event_ts = add_start_end_ts(event_ts , -8)

    # #during event (may have to be changed for future events)
    # elif(mode == 2):
    #     event_ts = add_start_end_ts(event_ts , 1)

    # #custom (delta will be non-zero in this case)
    # else:
    event_ts = add_start_end_ts(event_ts , delta)  

    # EVENT ISOLATION:
    isolated_event_df = pd.DataFrame()
    if filter_out_other_events:
        isolated_event_df = isolate_event(event_ts , time_gap_hours , selected_event , sub_event_dict)
    else:
        isolated_event_df = event_ts

    ## SUB-EVENT FILTERING:
    cleaned_sub_events = [s.replace(" ", "").strip().lower() for s in sub_event_dict[selected_event]]

    # cleaning the keys in the dictionary as it is later used for string matching in calc_event_spec_returns.
    cleaned_sub_event_filtering_dict = {
    re.sub(r'\s+', '', k).strip().lower(): v
    for k, v in sub_event_filtering_dict.items()
    }

    # event_df only contains the sub_events corresponding to the main event (selected_event), after isolation.
    event_df = isolated_event_df.loc[
        isolated_event_df['events']
        .astype(str)
        .str.replace(" ", "")
        .str.slice(0, -3)
        .str.strip()
        .str.lower()
        .isin(cleaned_sub_events)
    ]

    print("LENGTH 1:" , len(event_df))
    print(event_df.head(30))

    valid_timestamps = []
    timestamp_list = event_df['datetime'].unique().tolist()

    for ts in timestamp_list:
        temp_df = event_df[event_df['datetime'] == ts]
        flag = True

        for event , bounds in cleaned_sub_event_filtering_dict.items():
            
            # Filter rows in the current timestamp group that match this cleaned event
            match_mask = (
                temp_df['events']
                .str.replace('\xa0', '', regex=False)
                .str.replace(' ', '')
                .str.strip()
                .str.lower()
                .str.slice(0 ,-3) == event
            )
            
            filtered = temp_df[match_mask]

            #there is 1 row that matches the sub_event being considered
            if len(filtered) == 1:

                val = filtered['actual'].values[0] - (
                    filtered['consensus'].values[0] if not pd.isna(filtered['consensus'].values[0]) 
                    else filtered['forecast'].values[0]
                )

                if(pd.isna(bounds[0]) or pd.isna(bounds[1])):
                    print("Bounds NaN")

                if (val < bounds[0] or val > bounds[1]):
                    flag = False
                    break

            else:
                flag = False
                break

        if flag:
            valid_timestamps.append(ts)

    event_df = event_df[event_df['datetime'].isin(valid_timestamps)]
    event_df.dropna(inplace=True)

    if not event_df.empty:

        print("LENGTH 2:" , len(event_df['datetime'].unique().tolist()))

        cutoff_time = pd.to_datetime('2022-12-20 00:00:00-05:00', errors='coerce')
        event_df = event_df[event_df['start'] >= cutoff_time]

        print("LENGTH 3:" , len(event_df['datetime'].unique().tolist()))

        print(event_df[['actual' , 'consensus' , 'forecast' , 'datetime' , 'events']])

        final_df=pd.DataFrame()
        vol_ret = []
        abs_ret = []
        ret = []
        start_date = []
        end_date = []
        start_price = []
        end_price = []
        high = []
        low = []

        for end , start in zip(event_df['end'], event_df['start']):

            temp_df = ohcl_1h[(ohcl_1h['US/Eastern Timezone'] >= start) & (ohcl_1h['US/Eastern Timezone'] < end)] #equality removed for 'end'. Otherwise 1 extra hour in taken.
            entry_price = None # open price for the custom session made
            exit_price = None # close price for the custom session made
            maxi = None # Highest price during the custom session made
            mini = None # Lowest price during the custom session made
            
            if(temp_df.empty):
                vol_ret.append(np.nan)
                abs_ret.append(np.nan)
                ret.append(np.nan)
                start_date.append(np.nan)
                end_date.append(np.nan)
                start_price.append(np.nan)
                end_price.append(np.nan)
                high.append(np.nan)
                low.append(np.nan)

            else:
                entry_price = temp_df['Open'].iloc[0]
                exit_price = temp_df['Close'].iloc[-1]
                maxi = temp_df['High'].max()
                mini = temp_df['Low'].min()

                vol_ret.append((temp_df['High'].max() - temp_df['Low'].min())*16)
                abs_ret.append(abs(temp_df['Close'].iloc[-1] - temp_df['Open'].iloc[0])*16)
                ret.append((temp_df['Close'].iloc[-1] - temp_df['Open'].iloc[0])*16)
                start_date.append(temp_df['US/Eastern Timezone'].iloc[0])
                end_date.append(temp_df['US/Eastern Timezone'].iloc[-1])
                start_price.append(entry_price)
                end_price.append(exit_price)
                high.append(maxi)
                low.append(mini)

        final_df['Volatility Return'] = vol_ret
        final_df['Absolute Return'] = abs_ret
        final_df['Return'] = ret
        final_df['Start_Date'] = start_date
        final_df['End_Date'] = end_date
        final_df['Entry_Price'] = start_price
        final_df['Exit_Price'] = end_price
        final_df['High'] = high
        final_df['Low'] = low
        print(final_df.head())

        final_df.dropna(inplace=True)

        print("SELECTED EVENT: ", selected_event)
        print('No. of Data points: ' , len(final_df['Start_Date'].unique().tolist()))

        return (final_df , event_df)
    else:
        return (pd.DataFrame() , event_df)
        
#5.2 plot the event specific returns
def plot_event_spec_returns(final_df , selected_event):
        
    figures = {}

    for col in final_df.columns[:3]:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot histogram and KDE
        sns.histplot(final_df[col], kde=True, stat="density", linewidth=0, color="skyblue", ax=ax)
        sns.kdeplot(final_df[col], color="darkblue", linewidth=2, ax=ax)

        #number of instances.
        print(len(final_df[col]))  

        # Annotate histogram bars with bin edges (left-right) on top of each bar
        for patch in ax.patches:
            height = patch.get_height()
            if height == 0:
                continue  # skip bars with zero height
            left = patch.get_x()
            right = left + patch.get_width()
            x_center = left + patch.get_width() / 2
            label = f"{left:.2f} - {right:.2f}"
            ax.annotate(
                label,
                xy=(x_center, 0),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                fontsize=4,
                color='black'
            )

        # Statistics
        stats = final_df[col].describe()
        mean = stats['mean']
        std = stats['std']
        current_value = final_df[col].iloc[-1]
        current_date = final_df['Start_Date'].iloc[-1].date()

        zscore = (current_value - mean) / std if std != 0 else 0
        latest_percentile = percentileofscore(final_df[col].squeeze(), current_value , kind="rank").round(2)

        # Red dot just above x-axis
        _ , y_max = ax.get_ylim()
        dot_y = y_max * 0.02

        # Red dot and vertical line
        ax.plot(current_value, dot_y, 'ro', label='Current Value')
        ax.axvline(x=current_value, color='red', linestyle='dotted', linewidth=1)

        # Annotate with current value, z-score, percentile, and date
        annotation_text = (
            f"Date: {current_date}, "
            f"Value: {current_value:.2f}, "
            f"Z: {zscore:.2f}, "
            f"%ile: {latest_percentile}"
        )
        ax.annotate(
            annotation_text,
            xy=(current_value, dot_y),
            xytext=(current_value, y_max * 0.15),
            arrowprops=dict(facecolor='red', arrowstyle='->'),
            fontsize=9,
            fontweight='bold',
            color='red',
            ha='center'
        )

        # Add statistics box
        textstr = (
            f"Mean: {mean:.2f}\n"
            f"Std: {std:.2f}\n"
            f"Min: {stats['min']:.2f}\n"
            f"25%: {stats['25%']:.2f}\n"
            f"Median: {stats['50%']:.2f}\n"
            f"75%: {stats['75%']:.2f}\n"
            f"95%: {final_df[col].quantile(0.95):.2f}\n"
            f"99%: {final_df[col].quantile(0.99):.2f}\n"
            f"Max: {stats['max']:.2f}"
        )

        ax.text(0.75, 0.75, textstr, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"{col}")

        figures[col] = fig
        
    st.title("Distribution Analysis")
    col1, col2, col3 = st.columns(3)

    # Display each figure in a separate column
    with col1:
        st.pyplot(figures["Absolute Return"])
        st.write("**Absolute Return = [abs(close-open)]**")

    with col2:
        st.pyplot(figures["Return"])
        st.write("**Return = [close - open]**")

    with col3:
        st.pyplot(figures["Volatility Return"])
        st.write("**Volatility Return = [high - low]**")

    
# Setting up page configuration
st.set_page_config(
    page_title="FR Live Plots",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

tab = st.radio('Select a Tab' , ["Session and Volatility Returns for all sessions", 
                                "Latest X days of Volatility Returns for each session",
                                "Probability Matrix",
                                "Custom Normalised Returns",
                                "Event Specific Distro"])


# Defining GitHub Repo
repo_name='DistributionProject'
branch='main'
plots_directory="Intraday_data_files_stats_and_plots_folder"
plot_url_base=f"https://raw.githubusercontent.com/krishangguptafibonacciresearch/{repo_name}/{branch}/{plots_directory}/"

# Storing data in the form of links to be displayed later in separate tabs.
plot_urls=[]
intervals=[]
instruments=[]
return_types=[]

sessions=[]
latest_custom_days_urls=[]
for file in os.scandir(plots_directory):
    if file.is_file():
        if file.name.endswith('.png'):
            plotfile_content=file.name.split('_')
            plot_url=plot_url_base+file.name
            instrument=plotfile_content[0]
            interval=plotfile_content[1]
            return_type=plotfile_content[2]

            intervals.append(interval)
            instruments.append(instrument)
            plot_urls.append({
                "url": plot_url,
                "instrument": instrument,
                "interval": interval,
                "return_type": return_type,
                "stats_url": 
                (plot_url_base+f'{instrument}_{interval}_{return_type}_stats.csv').replace('Volatility', 'Volatility_Returns')
            })
        elif file.name.endswith('.csv') and 'latest_custom_days' in file.name:
            if 'stats' not in str(file.name):
                latest_custom_days_content=file.name.split('_')
                latest_custom_days_url=plot_url_base+file.name
                joined_session="_".join((latest_custom_days_content[0:-7:1]))
                spaced_session=" ".join(joined_session.split('_'))
                instrument=(latest_custom_days_content[-1])
                instrument=instrument.replace('.csv','')
                interval=latest_custom_days_content[-2]
                return_type=latest_custom_days_content[-4]

                sessions.append(spaced_session)
                latest_custom_days_urls.append({
                "url": latest_custom_days_url,
                'stats_url':plot_url_base+(file.name).split('.')[0]+'_stats.csv',
                "instrument": instrument,
                "interval": interval,
                "return_type": return_type,
                "session": [joined_session,spaced_session]
            })
            
# Storing unique lists to be used later in separate drop-downs
unique_intervals=list(set(intervals)) #Interval drop-down (1hr,15min,etc)
unique_instruments=list(set(instruments)) #Instrument/ticker drop-down (ZN, ZB,etc)
unique_sessions=list(set(sessions)) #Session drop-downs (US Mid,US Open,etc)
unique_versions=['Absolute','Up','Down','No-Version']#Version drop-downs for Probability Matrix
latest_days=[14,30,60,120,240,'Custom']
data_type = ['Non-Event' , 'All data']  #type of data to use when forming the Probability Matrix


# The  default option when opening the app
desired_interval = '1h'
desired_instrument='ZN'
desired_version='Absolute'


# Set the desired values in respective drop-downs.
# Interval drop-down
if desired_interval in unique_intervals:
    default_interval_index = unique_intervals.index(desired_interval)  # Get its index
else:
    default_interval_index = 0  # Default to the first element

# Instrument drop-down
if desired_instrument in unique_instruments:
    default_instrument_index = unique_instruments.index(desired_instrument)  # Get its index
else:
    default_instrument_index = 0  # Default to the first element

# Version drop-down
if desired_version in unique_versions:
    default_version_index = unique_versions.index(desired_version)  # Get its index
else:
    default_version_index = 0 # Default to the first element

# Create drop-down and display it on the left permanantly
x= st.sidebar.selectbox("Select Interval",unique_intervals,index=default_interval_index)
y= st.sidebar.selectbox("Select Instrument",unique_instruments,index=default_instrument_index)


#Define tabs:
if tab == "Session and Volatility Returns for all sessions":

        # Set title
        st.title("Combined Plots for all sessions")

        # Create checkboxes for type of return
        vol_return_bool = st.checkbox("Show Volatility Returns (bps)")
        return_bool = st.checkbox("Show Session Returns (bps)")

        
        # Store in session state
        st.session_state.x = x
        st.session_state.y = y

    
        # Get urls of the returns and volatility returns plot.
        filtered_plots = [plot for plot in plot_urls if plot["interval"] == x and plot["instrument"] == y]

        # Set volatility returns on 0th index and returns on 1st index. (False gets sorted first)
        filtered_plots = sorted(
            filtered_plots,
            key=lambda plot: (plot["return_type"] == "Returns", plot["return_type"])
        ) 

        # As per checkbox selected, modify the filtered_plots list.
    

        if vol_return_bool and return_bool:
            display_text='Displaying plots for all available Returns type.'
            return_type='Session_and_Volatility_Returns'

        elif vol_return_bool:
            display_text='Displaying plots for Volatility Returns only.'
            for index,fname in enumerate(filtered_plots):
                if 'Volatility' not in fname['return_type']:
                    filtered_plots.pop(index)
            return_type='Volatility_Returns'
        
        elif return_bool:
            display_text='Displaying plots for Session Returns only.'
            for index,fname in enumerate(filtered_plots):
                if 'Returns' not in fname['return_type']:
                    filtered_plots.pop(index)
            return_type='Session_Returns'
        
        else:
            filtered_plots=[]
            display_text=''
        st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)


        # Display plots and stats
        try:
            if filtered_plots:
                all_dataframes=[]
                tab1_sheet_names=[]
                image_url_list=[]
                tab1_image_names=[]
                for plot in filtered_plots:
                    caption = f"{plot['return_type'].replace('Returns', 'Returns Distribution').replace('Volatility', 'Volatility Distribution')}"
                    st.subheader(caption + ' Plot')
                    st.image(plot['url'],caption=caption,use_container_width=True)
                    st.subheader('Descriptive Statistics')
                    st.dataframe(
                        pd.read_csv(plot['stats_url']),
                        use_container_width=True
                    )

                    # Save Stats dataframes into a list
                    all_dataframes.append(pd.read_csv(plot['stats_url']))
                    tab1_sheet_names.append(caption+' Stats')

                    # Save images into a list
                    image_url_list.append(plot['url'])
                    tab1_image_names.append(f'{y}_{x}_{caption}')

                # Download Stats dataframes as Excel
                excel_file = download_combined_excel(
                    df_list=all_dataframes,
                    sheet_names=tab1_sheet_names,
                    skip_index_sheet=tab1_sheet_names
                )

                # Provide the Excel download link
                st.download_button(
                    label="Download Descriptive Statistics Data for selected Return type(s)",
                    data=excel_file,
                    file_name=f'{return_type}_{x}_{y}_stats.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Provide plots download link

                if "button_clicked" not in st.session_state:
                    st.session_state["button_clicked"] = False  # To track if the button is clicked
                    st.session_state["image_bytes_list"] = None  # To store downloaded images

                # Display the button
                if st.button("Download Image Plots"):
                    # Show the "Please wait..." message in red
                    st.session_state["button_clicked"] = True
                    wait_placeholder = st.empty()

                    # Display "Please wait..." in red
                    wait_placeholder.markdown("<span style='color: green;'>Please wait...</span>", unsafe_allow_html=True)

                    process_images(image_url_list)
                        
                    # Remove the "Please wait..." message
                    wait_placeholder.empty()
                # Handle the state when button is clicked and images are ready
                if st.session_state["image_bytes_list"] is not None:
                    st.markdown(
                        "<span style='color: white;'>(Following images are ready for download):</span>",
                        unsafe_allow_html=True
                    )
                    for img_byte, img_name in zip(st.session_state["image_bytes_list"], tab1_image_names):
                        st.download_button(
                            label=f"Download {img_name.split('_')[-1]} plot",
                            data=img_byte,
                            file_name=img_name + ".png",
                            mime="image/png"
                        )

            else:
                if vol_return_bool or return_bool:
                    st.write("No plots found for the selected interval and instrument.")
                else:
                    st.write('Please select Return type!')

        except FileNotFoundError as e:
            print(f'File not found: {e}. Please try again later.')

elif tab == "Latest X days of Volatility Returns for each session":
    
        st.title("Get Volatility Returns for custom days")
        
        # Use stored values from session state
        x = st.session_state.get("x", list(unique_intervals)[0])
        y = st.session_state.get("y", list(unique_instruments)[0])
        
        # Show the session dropdown
        selected_sessions = st.multiselect("Select Session", unique_sessions)

        if not selected_sessions:
            st.warning("Please select at least one session to continue.")
            st.stop()


        data_type = st.selectbox("Select which data to analyze" , ['All Data' , 'Non-event Data'])
        
        data = None
        intraday_data_folder = 'Intraday_data_files_pq'
        nonevent_data_folder = 'Intraday_data_files_processed_folder_pq'
        if(data_type == 'All Data'):
            for file in os.scandir(intraday_data_folder):
                if x in file.name and y in file.name:
                    data = pd.read_parquet(os.path.join(intraday_data_folder , file.name) , engine = 'pyarrow')
                    data['timestamp'] = data['US/Eastern Timezone']
                    data['session'] = data['timestamp'].apply(get_session)
                    print('Data being used: ', file.name)
        else:
            for file in os.scandir(nonevent_data_folder):
                if x in file.name and y in file.name and 'nonevents' in file.name:
                    data = pd.read_parquet(os.path.join(nonevent_data_folder , file.name) , engine = 'pyarrow')
                    print('Data being used: ',file.name)

        # Select number of days to analyse
        get_days=st.selectbox("Select number of days to analyse", latest_days,index=0)
        get_days_val=get_days

        if get_days_val=='Custom':
            enter_days=st.number_input(label="Enter the number of days:",min_value=1 , max_value = len(list(data['timestamp'].dt.date.unique())) , value = 10 , step=1)
            get_days_val=enter_days

        if(not data.empty):
            data = data[['timestamp' , 'session' , 'Adj Close' ,'Close' ,'High' , 'Low' , 'Open' , 'Volume' , 'US/Eastern Timezone']]
            data['date'] = data['timestamp'].dt.date
            filtered_data = data[data['session'].isin(selected_sessions)]

            print("Required Data: " , filtered_data.tail())

            daily_vol = filtered_data.groupby('date').agg({'High': 'max', 'Low': 'min'})
            daily_vol['volatility'] = daily_vol['High'] - daily_vol['Low']

            # Global Z-score 
            vol_mean_all = daily_vol['volatility'].mean()
            vol_std_all = daily_vol['volatility'].std()
            daily_vol['zscore_all'] = (daily_vol['volatility'] - vol_mean_all) / vol_std_all

            # Rolling z-score (last X days up to each day)
            daily_vol['zscore_last_X_days'] = daily_vol['volatility'].rolling(get_days_val).apply(
                lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0,
                raw=False
            )

            # display the volatility df on the screen.
            daily_vol = daily_vol.reset_index()
            # cutoff_date = daily_vol['date'].max() - pd.Timedelta(days = get_days_val)
            # daily_vol = daily_vol[daily_vol['date'] >= cutoff_date]
            daily_vol = daily_vol[-get_days_val:]
            display_df = daily_vol[['date' ,'volatility' , 'zscore_last_X_days' , 'zscore_all']]
            st.subheader(f"Volatility Returns for Latest {get_days_val} day(s) of the sessions: {selected_sessions}")
            st.dataframe(display_df , use_container_width=True)


            #descriptive stats
            percentiles = [0.10, 0.25, 0.50, 0.75, 0.95, 0.99]
            percentile_values = daily_vol['volatility'].quantile(percentiles).round(2)
            percentile_dict = {f"{int(p*100)}%": v for p, v in percentile_values.items()}

            stats_dict = {
                'latest_date': daily_vol.index[-1],
                'count': len(daily_vol['date'].unique()) if 'date' in daily_vol.columns else len(daily_vol),
                'latest_vol': daily_vol['volatility'].iloc[-1],
                'zscore_all': daily_vol['zscore_all'].iloc[-1],
                'zscore_last_N': daily_vol['zscore_last_X_days'].iloc[-1],
                'mean_vol': vol_mean_all,
                'std_vol': vol_std_all,
                'skew_val': skew(daily_vol['volatility']),
                'kurt_val': kurtosis(daily_vol['volatility']),
                'min_val': daily_vol['volatility'].min(),
                'max_val': daily_vol['volatility'].max()
            }

            stats_dict.update(percentile_dict)
            
            #converting the stats to a df
            stats_df = pd.DataFrame([stats_dict])

            # display the stats_df on the screen.
            st.subheader("Descriptive Statistics")
            st.dataframe(stats_df,use_container_width=True)

            # plotting graph
            plt.figure(figsize=(10, 5))
            sns.kdeplot(daily_vol['volatility'], fill=True, linewidth=2, label="Volatility KDE")

            # Red dot for latest volatility
            latest_vol_val = daily_vol['volatility'].iloc[-1]
            latest_vol_date = daily_vol['date'].iloc[-1]
            latest_vol_z_score = daily_vol['zscore_last_X_days'].iloc[-1]
            plt.scatter(daily_vol['volatility'].iloc[-1], 0, color='red', s=100, zorder=5, label='Latest Day')
            plt.axvline(x=latest_vol_val, color='red', linestyle='--', linewidth=1)

            annotation_text = f"{latest_vol_date} | Val: {latest_vol_val:.2f} | Z-score: {latest_vol_z_score:.1f}"

            # Annotate above the axis
            plt.annotate(
                annotation_text,
                xy=(latest_vol_val, 0),
                xytext=(latest_vol_val, 0.25),  # Raise it above the x-axis
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red',
                fontsize=10,
                fontweight='bold'
            )

            # Labels and styling
            plt.title(f'Daily Volatility Returns (High - Low) for {", ".join(selected_sessions)} Sessions')
            plt.xlabel('Volatility')
            plt.ylabel('Density')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt) 

            # Combine the DataFrames into an Excel file
            excel_file = download_combined_excel(
                df_list=[daily_vol , stats_df],
                sheet_names=['Volatility Returns', 'Descriptive Statistics'],
                skip_index_sheet=['Volatility Returns'],
            )

            # Provide the download link
            st.download_button(
                label="Download Returns and Statistical Data",
                data=excel_file,
                file_name=f'Latest_{get_days_val}_Volatility_Returns_{x}_{y}.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
                
        else:
            st.write("No data found for the selected session.")
        

elif tab == "Probability Matrix":
        try:
            st.title("Probability Matrix (Unconditional)")
            # Use stored values from session state
            x = st.session_state.get("x", list(unique_intervals)[0])
            y = st.session_state.get("y", list(unique_instruments)[0])
            if 'h' in x:
                # Show the version dropdown
                version_value = st.selectbox("Select Version",unique_versions,index=default_version_index)

                data_type = st.selectbox("Select type of data to use", data_type , index=default_version_index)

                # Select bps to analyse
                enter_bps=st.number_input(label="Enter the number of bps:",min_value=0.0, step=0.5)
                st.caption("Note: The value must be a float and increases in steps of 0.5. Eg 1, 1.5, 2, 2.5, etc") 
                st.caption("The probability matrix rounds offs any other bps value into this format in the output.")

                # Select number of hours to analyse
                enter_hrs=st.number_input(label="Enter the number of hours:",min_value=1, step=1)
                st.caption("Note: The value must be an integer and increase in steps of 1. Eg 1, 2, 3, 4, etc.")
            
                # Get the probability matrix
                v=version_value
                
                prob_matrix_dic=GetMatrix(enter_bps,enter_hrs,x,y, data_type , version=version_value)
                st.subheader(f"Probability of bps ({v})  > {abs(enter_bps)} bps within {enter_hrs} hrs")

                # Store > probability in a small dataframe
                prob_df=pd.DataFrame(columns=['Description','Value'],
                            data=[[f'Probability of bps ({v})  > {abs(enter_bps)} bps within {enter_hrs} hrs',
                                str(round(prob_matrix_dic[v]['>%'],2))+'%'] ]
                )
                # Store <= probability in the dataframe
                prob_df.loc[len(prob_df)] = [f'Probability of bps ({v})  <= {abs(enter_bps)} bps within {enter_hrs} hrs',
                                            str(round(prob_matrix_dic[v]['<=%'],2))+'%']
                
                # Display the probability dataframe
                st.dataframe(prob_df,use_container_width=True)

                # Display the probability plots
                st.subheader(f"Probability Plot for {enter_bps} bps ({v}) movement in {enter_hrs} hrs")
                st.pyplot(prob_matrix_dic[v]['Plot'])

                st.subheader("Probability Plot for max(high-open , open-low)")
                st.pyplot(prob_matrix_dic['OH_OL_plot']['Plot'])

                # Display the probability matrix
                my_matrix=prob_matrix_dic[v]['Matrix']
                my_matrix.columns=[str(i)+' hr' for i in my_matrix.columns]
                my_matrix.index=[str(i)+' bps' for i in my_matrix.index]
                st.subheader(f"Probability Matrix of Pr(bps ({v}) >)")
                st.dataframe(my_matrix)


                # Combine the DataFrames into an Excel file
                my_matrix_list=[]
                my_matrix_ver=[]
                for ver in list(prob_matrix_dic.keys()):
                    if(ver != 'OH_OL_plot'):
                        my_matrix_list.append(prob_matrix_dic[ver]['Matrix'])
                        my_matrix_ver.append(f'{ver} bps Probability Matrix (> form)')
            
                excel_file = download_combined_excel(
                    df_list=my_matrix_list,
                    sheet_names=my_matrix_ver,
                    skip_index_sheet=[]
                )

                # Provide the download link for plots
                valid_keys = []  #first remove the OH_OL_plot key.
                for ver in prob_matrix_dic.keys():
                    if(ver != "OH_OL_plot"):
                        valid_keys.append(ver)
                st.download_button(
                    label=f"Download the Probability Matrices for version(s): bps {", bps ".join(list(valid_keys))}",
                    data=excel_file,
                    file_name=f"Probability Matrix_{'_'.join(my_matrix_ver)}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Provide plots download link
                if "tab3_button_clicked" not in st.session_state:
                    st.session_state["tab3_plots_button_clicked"] = False  # To track if the button is clicked
                    st.session_state["tab3_plots_ready"] = None 

                # Display the button
                if st.button("Download Image Plots",key='tab3_button'):
                    # Show the "Please wait..." message in red
                    st.session_state["tab3_plots_button_clicked"] = True
                    wait_placeholder2 = st.empty()

                    # Display "Please wait..." in red
                    wait_placeholder2.markdown("<span style='color: green;'>Please wait...</span>", unsafe_allow_html=True)
            
                    
                    # Handle the state when button is clicked and images are ready
                    if st.session_state["tab3_plots_ready"] is not None:
                        st.markdown(
                            "<span style='color: white;'>(Following images are ready for download):</span>",
                            unsafe_allow_html=True
                        )
        
                    for ver,_ in prob_matrix_dic.items():
                        if(ver != 'OH_OL_plot'):
                            my_img_data = download_img_via_matplotlib(prob_matrix_dic[ver]['Plot'])
                            st.download_button(
                                label=f"Download the Probability Plots for version: bps {ver}",
                                data=my_img_data,
                                file_name=f"Probability Matrix_{ver}.png",
                                mime="image/png"
                            )
                    
                    # Remove the "Please wait..." message
                    wait_placeholder2.empty()
                
            else:
                st.write("Please select 1h interval.")
        except:
            display_text='1h interval data unavailable for the current ticker.'
            st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

elif tab == "Custom Normalised Returns":
            try:
                # Protected tab
                # Add password
                PASSWORD = "distro" 

                # Initialize authentication state
                if "authenticated" not in st.session_state:
                    st.session_state.authenticated = False

                if not st.session_state.authenticated:
                    st.header("This tab is Password Protected🔒")
                    password = st.text_input("Enter Password:", type="password")
                    
                    if st.button("Login"):
                        if password == PASSWORD:
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("Incorrect password. Try again.")
                else:
                    st.header("Authorised ✅")
                    st.write("This tab contains sensitive information.")
                    
                    if st.button("Logout"):
                        st.session_state.authenticated = False
                        st.rerun()
                    

                if st.session_state.authenticated==True:
                    # Use stored values from session state
                    x = st.session_state.get("x", list(unique_intervals)[0])
                    y = st.session_state.get("y", list(unique_instruments)[0])

                    st.title("Custom Filtering")

                    # Default sessions:
                    
                    # Show the version dropdown
                    version_value = st.selectbox("Select Version",unique_versions.copy(),index=default_version_index,
                                                key='tab4_v')

                    # Select bps to analyse
                    enter_bps=st.number_input(label="Enter the Observed movement in bps:",min_value=0.00,key='tab4_bps')

                    # Select Multiple Sessions

                    # Add custom session via button
                    default_text=f'Distribution of bps ({version_value}) Returns {y} with returns calculated for every {x}'
                    finalname=default_text
                    final_list=[]
                    
                    filter_sessions=False
                    
                    # Not include intervals
                    if 'd' not in x:
                        st.subheader('Add Custom Session')
                        tab4check=st.checkbox(label='Add Custom Session',key='tab4check')

                        if tab4check:
                            # Add Checkbox to filter by starting day
                            tab4check1=st.checkbox(label='Calculate Custom Time Difference',key='tab4check1')
                            if tab4check1:
                                # Date inputs
                                start_date = st.date_input(label="Start Date (YYYY/MM/DD)", value=datetime.today().date())
                                end_date = st.date_input(label="End Date (YYYY/MM/DD)", value=datetime.today().date())

                                # Time inputs
                                start_time = st.time_input(label="Start Time (HH:MM)",value='now',help='Directly Type Time in HH:MM')
                                end_time = st.time_input(label="End Time (HH:MM)",value='now',help='Directly Type Time in HH:MM')
                            
                                # Combine date and time into datetime objects
                                start_datetime = datetime.combine(start_date, start_time)
                                end_datetime = datetime.combine(end_date, end_time)

                                # Calculate time difference
                                time_diff = end_datetime - start_datetime

                                # Extract hours and minutes
                                hours, remainder = divmod(time_diff.total_seconds(), 3600)
                                minutes = remainder / 60

                                display_text1=(f"Time Difference: {int(hours)} hours and {int(minutes)} minutes")
                                display_text2=(f"Approx Difference (Hrs): {round(hours+minutes/60,1)} hours")
                                display_text3=(f"Approx Difference (Mins): {int(hours*60+minutes)} minutes")
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text1}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text2}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text3}</p>", unsafe_allow_html=True)

                            # 1. Select Start time in ET
                            enter_start=st.number_input(label="Enter the start time in ET",min_value=0, max_value=23, step=1)
                            st.caption("Note: The value must be an integer and increase in steps of 1. Eg 1, 2, 3, 4, etc.")
                            

                            # 2. Select number of hours to analyse post the start time
                            enter_hrs=st.number_input(label=f"Enter the time (multiple of {x}) to be searched post the selected time",min_value=0, step=1)
                            st.caption("Note: The value must be an integral multiple of the interval selected")


                            # Add Checkbox to filter by starting day
                            tab4check2=st.checkbox(label='Filter by Starting Day',key='tab4check2')
                            day_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                            
                            # Add Selectbox to select the starting day
                            if tab4check2==True:
                                enter_start_day=st.selectbox("Select Starting Day",day_list,index=0,
                                                key='tab4_sd')
                            else:
                                enter_start_day=""


                        
                        # Combine default and custom time filters. filter_sessions1=default, filter_sessions2=custom
                            filter_sessions1=[]
                            filter_sessions2=[]
                            filter_sessions1.append((enter_start,enter_hrs,enter_start_day))
                    
                        # Combine the two
                            filter_sessions=list(set(filter_sessions1+filter_sessions2))


                    # Give the name to include ticker,interval,time,day,start_date and end_date.
                    if filter_sessions==False:
                        filename=default_text
                    else:
                        mysession=f'{filter_sessions[0][2]} {filter_sessions[0][0]} ET to {filter_sessions[0][0]} ET+{filter_sessions[0][1]}{x[-1]}'
                        finalname=f'{default_text} for session:{mysession}'

                    # Select the dataframe for Hour interval
                    selected_df=custom_filtering_dataframe.get_dataframe(x,y,'Intraday_data_files_pq')

                    # Extract start and end dates
                    finalcsv=selected_df.copy()
                    finalcsv.index=finalcsv[finalcsv.columns[-1]]
                    finalcsv.drop_duplicates(inplace=True)
                    finalcsv.dropna(inplace=True,how='all') 
                    finalcsv.sort_index(inplace=True)
                    finalcsv = finalcsv.loc[~finalcsv.index.duplicated(keep='last')]
                    finalstart=str(finalcsv.index.to_list()[0])[:10]
                    finalend=str(finalcsv.index.to_list()[-1])[:10]


                    if filter_sessions:
                        # Filter the dataframe as per selections
                        filtered_df=custom_filtering_dataframe.filter_dataframe(selected_df,
                                                                                filter_sessions,
                                                                                day_dict="",#time_day_dict,
                                                                                timezone_column='US/Eastern Timezone',
                                                                                target_timezone='US/Eastern',
                                                                                interval=x,
                                                                                ticker=y)
                        finalname+=f' for dates:{finalstart} to {finalend}'
                        # Stats and Plots
                        stats_plots_dict=custom_filtering_dataframe.calculate_stats_and_plots(filtered_df,
                                                                            finalname,
                                                                            version=version_value,
                                                                            check_movement=enter_bps,
                                                                            interval=x,
                                                                            ticker=y,
                                                                            target_column='Group')

                    else:
                        finalname=f'{default_text} for dates:{finalstart} to {finalend}'
                        filtered_df=custom_filtering_dataframe.filter_dataframe(selected_df,
                                                                                "",
                                                                                "",
                                                                                'US/Eastern Timezone',
                                                                                'US/Eastern',
                                                                                x,
                                                                                y)
                        if(filtered_df.empty):
                            print('Empty')
                        else:
                            print("FDF" , filtered_df.head())
                        # Stats and Plots
                        stats_plots_dict=custom_filtering_dataframe.calculate_stats_and_plots(filtered_df,
                                                                            finalname,
                                                                            version=version_value,
                                                                            check_movement=enter_bps,
                                                                            interval=x,
                                                                            ticker=y,
                                                                            target_column='US/Eastern Timezone')

        
                    
                    # Add Widgets:
                    # Dataframe
                    st.subheader('Filtered Dataframe')
                    st.text(f'Ticker: {y}')
                    st.text(f'Interval: {x}')
                    st.text(f'Dates: {finalstart} to {finalend}')
                    # if filter_sessions==False:
                    #     session_text="None"
                    # else:
                    #     session_text=f'Start Time:{filter_sessions[0]}, Start Day:{filter_sessions[2]}, Filter for next {filter_sessions[1]} units post '
                    # st.text(f'Filters Applied: {session_text}')
                    st.dataframe(filtered_df,use_container_width=True)


                    # Display the  stats dataframe
                    stats_df=stats_plots_dict['stats']
                    st.dataframe(stats_df,use_container_width=True)

                    # Store > probability in a small dataframe
                    prob_df=pd.DataFrame(columns=['Description','Value'],
                                data=[[f'Probability of bps ({version_value})  > {abs(enter_bps)}',
                                    str(round(stats_plots_dict['%>'],2))+'%'] ]
                    )

                    # Store <= ZScore
                    prob_df.loc[len(prob_df)] =[f'Probability of bps ({version_value})  <= {abs(enter_bps)}',
                                    str(round(stats_plots_dict['%<='],2))+'%']
                    
                    prob_df.loc[len(prob_df)] =[f'ZScore for ({version_value}) bps <=  {enter_bps} bps',
                                    str((stats_plots_dict['zscore<=']))]
                

                    # Display the probability dataframe
                    st.dataframe(prob_df,use_container_width=True)


                    # Display the probability plot
                    st.subheader(f"Probability Plot for {enter_bps} bps ({version_value}) movement")
                    st.pyplot(stats_plots_dict['plot'])
                

                    # Combine the DataFrames into an Excel file (Convert datetime values to text)
                    filtered_df[filtered_df.columns[-3]]=filtered_df[filtered_df.columns[-3]].astype(str) # Datetime column
                    my_matrix_list=[filtered_df,
                                    prob_df,
                                    stats_df]
                    my_matrix_ver=[f'{x}_{y}_{finalstart} to {finalend}','Probability','Descriptive Statistics']
                
                    excel_file = download_combined_excel(
                        df_list=my_matrix_list,
                        sheet_names=my_matrix_ver,
                        skip_index_sheet=[]
                    )

                    # Provide the download link for plots
                    st.download_button(
                        label="Download Excels",
                        data=excel_file,
                        file_name=f"Probability_Stats_Excel_{finalname}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    my_img_data= download_img_via_matplotlib(stats_plots_dict['plot'])
                    st.download_button(
                            label=f"Download the Probability Plots",
                            data=my_img_data,
                            file_name=f"Probability Plot.png",
                            mime="image/png"
                        )
            except UnboundLocalError as uble:
                display_text=f'{y} Data unavailable for {x} interval.'
                st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

            except Exception as e:
                display_text='Some error occured. Please try some other parameters and re-run.'
                st.text(e)
                st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

elif tab == "Event Specific Distro":
        
    events = ['CPI', 'PPI', 'PCE Price Index', 'Non Farm Payrolls', 'ISM Manufacturing PMI', 'ISM Services PMI',
              'S&P Global Manufacturing PMI Final', 'S&P Global Services PMI Final', 'Michigan',
              'Jobless Claims' , 'ADP' , 'JOLTs' , 'Challenger Job Cuts' , 'Fed Interest Rate Decision' , 
              'GDP Price Index QoQ Adv' , 'Retail Sales' , 'Fed Press Conference', 'FOMC Minutes']
    
    sub_event_dict = {
        "CPI": ['Inflation Rate MoM' , 'Inflation Rate YoY' , 'Core Inflation Rate MoM' , 'Core Inflation Rate YoY' , 'CPI' , 'CPI s.a'],
        "PPI": ['Core PPI MoM' , 'Core PPI YoY' , 'PPI MoM' , 'PPI YoY'],
        "PCE Price Index": ['Core PCE Prices QoQ' , 'PCE Prices QoQ' , 'PCE Price Index MoM' , 'PCE Price Index YoY' , 'Core PCE Price Index MoM' , 'Core PCE Price Index YoY'],
        "Non Farm Payrolls": ['Non Farm Payrolls' , 'Unemployment Rate' , 'Average Hourly Earnings MoM' , 'Average Weekly Hours' , 'Government Payrolls' , 'Manufacturing Payrolls' , 'Nonfarm Payrolls Private' , 'Participation Rate'],
        "ISM Manufacturing PMI": ['ISM Manufacturing PMI' , 'ISM Manufacturing New Orders' , 'ISM Manufacturing Employment'],
        "ISM Services PMI": ['ISM Services PMI' , 'ISM Services New Orders' , 'ISM Services Employment' , 'ISM Services Business Activity' , 'ISM Services Prices'],
        'S&P Global Manufacturing PMI Final': ['S&P Global Manufacturing PMI Final'], 
        'S&P Global Services PMI Final': ['S&P Global Services PMI Final'],
        'Michigan': ['Michigan Consumer Sentiment Final' , 'Michigan Consumer Sentiment Prel'],
        'Jobless Claims': ['Initial Jobless Claims' , 'Continuing Jobless Claims'], 
        'ADP': ['ADP Employment Change'], 
        'JOLTs': ['JOLTs Job Openings' , 'JOLTs Job Quits'], 
        'Challenger Job Cuts': ['Challenger Job Cuts'], 
        'Fed Interest Rate Decision': ['Fed Interest Rate Decision'] , 
        'GDP Price Index QoQ Adv': ['GDP Price Index QoQ Adv' , 'GDP Growth Rate QoQ Adv'] , 
        'Retail Sales': ['Retail Sales MoM' , 'Retail Sales YoY' , 'Retail Sales Ex Autos MoM'] , 
        'Fed Press Conference': ['Fed Press Conference'], 
        'FOMC Minutes': ['FOMC Minutes']
    }
    
    selected_event = st.selectbox("Select an event:" , events)

    selected_sub_events = st.multiselect("Select a sub-event to condition it on:" , sub_event_dict[selected_event])

    sub_event_filtering_dict = {}   # dictionary containing the sub_events as keys and the upper and lower bound of filtering as the values.
    for sub_event in selected_sub_events:

        col1, col2 = st.columns([2, 2])
        upper_bound = None
        lower_bound = None

        with col1:
            lower_bound = st.number_input(f"Pls give the lower bound for the filtering range for {sub_event}" , step = 0.0001 , format="%.6f")
        with col2:
            upper_bound = st.number_input(f"Pls give the upper bound for the filtering range for {sub_event}" , step = 0.0001 , format="%.6f")

        sub_event_filtering_dict[sub_event] = [lower_bound, upper_bound]
    
    # For isolated event distribution.
    filter_isolated = st.checkbox(
    "Exclude events when there is another event announced x hours prior. (only events in the dropdown)",
    help="Only include event instances that have no other events in the surrounding time window."
    )

    #number of hours to check for the isolation of event.
    window_hrs = 0
    if(filter_isolated):
        window_hrs = st.number_input(label="Choose x:",min_value=1)
        st.text('x only takes on integer values')

    # all event timestamps
    for file in os.scandir("Intraday_data_files_processed_folder_pq"):
        if(file.name.endswith('.csv') and 'EconomicEventsSheet' in file.name and 'target' in file.name):
            all_event_ts = pd.read_csv(file.path)

    # timestamp is already timezone aware, so no need to .dt.tz_convert()
    all_event_ts['datetime'] = pd.to_datetime(all_event_ts['datetime'], errors='coerce')

    # finding the price movements:
    repo_name = "DistributionProject"
    branch = "main"
    plots_directory2 = "Intraday_data_files"

    # Regular expression to match file pattern. Has to be used since the file name changes.
    pattern = re.compile(r"Intraday_data_ZN_1h_2022-12-20_to_(\d{4}-\d{2}-\d{2})\.parquet")
    ohcl_1h = pd.DataFrame()

    for file in os.scandir('Intraday_data_files_pq'):
        if file.is_file():
            match = pattern.match(file.name)
            if match:
               print("File used:" , file.name)
               ohcl_1h = pd.read_parquet(os.path.join("Intraday_data_files_pq" , file.name) , engine = 'pyarrow')

    # convert US/Eastern Timezone from string data type to a [datetime , ET] datatype. (str --> UTC --> ET)
    # Datetime col has strings. so first convert that to UTC datetime.
    ohcl_1h['US/Eastern Timezone'] = pd.to_datetime(ohcl_1h.index,errors='coerce',utc=True) 
    all_event_ts['datetime'] = pd.to_datetime(all_event_ts['datetime'] , errors = 'coerce' , utc = True)

    #make it timezone aware.
    ohcl_1h['US/Eastern Timezone'] = ohcl_1h['US/Eastern Timezone'].dt.tz_convert('US/Eastern')
    all_event_ts['datetime'] = all_event_ts['datetime'].dt.tz_convert('US/Eastern') 


    # For analysis of custome time before or after an event.
    delta = st.number_input("Enter the number of hours:", min_value=-1000, max_value=1000 , value=1, step=1,
                          help="Enter custom number of integer hours to analyse. Positive input will analyze after the event and negative input will analyze before the event.")

    final_df = None
    filtered_event_data = None

    # calling the helper functions to populate the final_df and plot the required columns.
    final_df , filtered_event_data = calc_event_spec_returns(selected_event, all_event_ts, ohcl_1h , delta, filter_isolated , window_hrs , sub_event_filtering_dict , sub_event_dict)

    if not final_df.empty:
        plot_event_spec_returns(final_df , selected_event)

        ## making final_df available as an excel file.
        output = BytesIO()

        final_df['Start_Date'] = final_df['Start_Date'].dt.tz_localize(None)
        final_df['End_Date'] = final_df['End_Date'].dt.tz_localize(None)
        # filtered_event_data['datetime'] = filtered_event_data['datetime'].dt.tz_localize(None)


        # Write the DataFrame to the buffer as an Excel file
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Price Movt')
            # filtered_event_data.to_excel(writer , index=False, sheet_name='Event data')

        # Move the buffer's pointer to the beginning
        output.seek(0)

        # Streamlit download button
        st.download_button(
            label=f"📥 Download Data",
            data=output,
            file_name='EventSpecificData.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.text('The above file is for:')

        text_output = f"Event: {selected_event} when actual - expected of the sub-events satisfy:\n"
        for event, bounds in sub_event_filtering_dict.items():
            text_output += f"    {event}: {bounds}\n"
        st.text(text_output)

        st.text(f'Number of such instances: {len(final_df['Start_Date'].unique().tolist())}')
        # if(not custom and not filter_isolated):
        #     st.text(f'Durataion: {dur}')
        # else:
        st.text(f'Time: {delta} hrs relative to the event')
        if(filter_isolated):
            st.text(f'Event instances where there are other events in a window of ± {window_hrs} around the selected event are excluded.')
        st.text('''We use hourly data to plot ZN reaction graphs, with 8:00 ET serving as the standard reference point for how each event hour is treated.

        For events released between 8:01 ET and 8:56 ET, the immediate reaction is measured using the 8:00-9:00 ET candle. Since the event occurs partway through that hour, the post-release price action captured reflects less than a full hour of reaction. If the pre-event window covers the 8 hours prior to the release, it spans 12:00-8:00 ET, which excludes a portion of the hour immediately preceding the release.

        For events released between 8:56 ET and 8:59 ET, the immediate reaction is captured using the 9:00-10:00 ET candle. Any price movement in the final seconds before 9:00—such as a release at 8:59:50 ET—is not reflected in the immediate reaction distribution, as it falls outside the defined post-event window. If the pre-event window again covers the 8 hours prior to the release, it would still span 12:00-8:00 ET, thereby also excluding some of the time immediately before the release.

        We track the number of instances where rounding the release time either down (to 8:00-9:00) or up (to 9:00-10:00) affects how the immediate market response is measured:

        Number of rounded down occurrences: x

        Number of rounded up occurrences: y

        Once a sufficient amount of 1-minute data is available, we will transition to that granularity to more accurately capture market reactions to economic releases.''')
    else:
        st.text("No data available for the specified filters")
