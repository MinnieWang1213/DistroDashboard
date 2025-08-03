import pandas as pd
from preprocessing import ManipulateTimezone
import os

#for converting the strings like $1.2B to numeric values. 
# Used for conditional filtering (hotter/colder than exprected) in tab 5
def convert_shorthand(x):
    if pd.isna(x):
        return x

    x = str(x).strip().upper()
    x = x.replace(',', '').replace('$', '')

    # Remove any repeated leading dashes or symbols like "--", "-$", etc.
    while x.startswith('--') or x.startswith('-$') or x.startswith('$$'):
        x = x[1:]

    # Now handle suffix multipliers
    multiplier = 1
    if x.endswith('M'):
        multiplier = 1_000_000
        x = x[:-1]
    elif x.endswith('B'):
        multiplier = 1_000_000_000
        x = x[:-1]
    elif x.endswith('K'):
        multiplier = 1_000
        x = x[:-1]
    elif x.endswith('%'):
        multiplier = 0.01
        x = x[:-1]

    try:
        return float(x) * multiplier
    except ValueError:
        return pd.to_numeric(x, errors='coerce')

    
class Events:
    """Combines Events from Economic Events sheet and converts the timestamp to US/Eastern.
    """
    def __init__(self,excel,tier_dic={},flag_dic={},**kwargs):
        self.excel=excel
        self.sheets_dic=pd.read_excel(self.excel,sheet_name=None)   # reads and saves the excel sheet in the form of a dictionary. key = sheet name and value is the table in the sheet as a df.
        self.flag_dic=flag_dic
        self.new_events_folder=kwargs.get('new_events_folder')
        self.add_new_events_dic=kwargs.get('add_new_events_dic')
        self.change_tiers=kwargs.get('change_tiers')

        if tier_dic=={}:
            tier_sheet='NA'
            for sheet in self.sheets_dic.keys():
                if 'tier' in str(sheet).strip().lower() or 'tiers' in str(sheet).strip().lower():
                    tier_sheet=sheet
                    break
            if tier_sheet=='NA':
                raise AttributeError('No Tier Data found. Please add a Tier Sheet with Events and Tier value.')
            
            temp_dic=(self.sheets_dic[tier_sheet].to_dict())
            events_tiers=list(temp_dic.values())[0:2]
            allevents=events_tiers[0]
            alltiers=events_tiers[1]
            for index1,index2 in zip(allevents,alltiers):
                if index1==index2:
                    tier_dic[allevents[index1]]=alltiers[index1]
            self.tier_dic=tier_dic
        else:
            self.tier_dic=tier_dic
        print(self.tier_dic)

        self.combined_excel=self.merge_sheets(self.sheets_dic,self.tier_dic,self.flag_dic,change_tiers=self.change_tiers)

    def __str__(self):
        return f'The selected events excel contains the following sheets: {list(self.sheets_dic.keys())}'
    
    def merge_sheets(self,sheets_dic,tier_dic,flag_dic,change_tiers):
        sheets_list=[]
        for key in (sheets_dic):
            years=[str(i) for i in range(2015,2025)]
            if key in (years):
                sheets_list.append(sheets_dic[key])
        merged_sheet =pd.concat(sheets_list[::-1],ignore_index=True)
        formatted_merged_sheet=self.format_sheet(merged_sheet,change_tiers,tier_dic,flag_dic)
        return formatted_merged_sheet
    
                
    def format_sheet(self,df,change_tiers,tier_dic={},flag_dic={}):
        
        # Convert TIME->DATETIME
        df.columns = df.columns.str.strip().str.lower()
        df=df.dropna(how='all')
        df.loc[:,'time']=df.loc[:,'time'].fillna(method='ffill')
        
        # Identify rows with a full date
        df.loc[:,'only_date'] = df.loc[:,'time'].str.len()>=8
        
        # Propagate dates downward
        df.loc[:,'date'] = df.loc[:,'time'].where(df['only_date']).ffill()
        df.loc[:,'date']=df.loc[:,'date'].apply(lambda d: " ".join(d.split(' ')[1:]))
        df=df[df['only_date']==False]
        
        # Combine propagated date with time
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'].astype(str).str.split().str[-1], errors='coerce'
        )

        # Since sheet is as per IST timezone
        df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')   

        # Add new events
        for tz_key in self.add_new_events_dic:
            df=self.append_new_events(df[['datetime','events','tier' , 'actual' , 'consensus' , 'previous' , 'forecast']],events=self.add_new_events_dic[tz_key],timezone=tz_key) 
       
        df['year']=(df['datetime'].astype(str).str.split().str[0]).str.split('-').str[0]

        if change_tiers==True:
            df = df[['datetime','events','tier' , 'actual' , 'consensus' , 'previous' , 'forecast']]
            # Add New Tiers
            df['tier']=self.assign_tier(df,tier_dic)
        
        else: #only apply tiers for nan values
            df.loc[:,'tier'] = df.loc[:,'tier'].fillna(df.apply(lambda row: self.assign_tier_helper(row, tier_dic), axis=1))
  
        df=df[['datetime','events','tier' , 'actual' , 'consensus' , 'previous' , 'forecast']]
        df['actual'] = df['actual'].apply(convert_shorthand)
        df['previous'] = df['previous'].apply(convert_shorthand)
        df['consensus'] = df['consensus'].apply(convert_shorthand)
        df['forecast'] = df['forecast'].apply(convert_shorthand)

        # Add flags (IND)
        finaldf=self.assign_flag(df,flag_dic)

        finaldf.dropna(inplace=True,how='all')
        return finaldf

    def append_new_events(self,old_events_df,events:list,timezone:str):
        try:
            # Finding the new events file
            new_e=pd.DataFrame()
            for file in os.scandir(self.new_events_folder):    
                for event in events:
                    if ( file.is_file() and (file.name.endswith('.csv')) and (event in str(file.name)) and (timezone in str(file.name))):   #add file.name.endswith('.xlsx') in the future if needed.
                        
                        if file.name.endswith('csv'):
                            new_e=pd.read_csv(os.path.join(self.new_events_folder,file.name))
                        else:
                            new_e=pd.read_excel(os.path.join(self.new_events_folder,file.name))
                      
            # Changing column name to suit with old events file
                        
                        new_e.columns=new_e.columns.str.lower()
                        new_e=new_e[['datetime','events','tier' , 'actual' , 'consensus' , 'previous' , 'forecast']]
                        new_e['tier']=new_e['tier'].astype(int)
 
            # Appending the data
                        finaldf=pd.concat([old_events_df,new_e],ignore_index=True,sort=True)
                        return finaldf

        except Exception as e:
            raise ValueError(e)


    def assign_tier(self, finaldf,tier_dic):
        finaldf['tier'] = finaldf.apply(lambda row: self.assign_tier_helper(row,tier_dic), axis=1)
        return finaldf['tier']
        
    def assign_tier_helper(self, row,tier_dic):
        myevent = str(row['events']).strip().lower()
        for event_key in tier_dic:
            if str(event_key).strip().lower() in myevent:
                return tier_dic[event_key]
        return 4  # Default value if no match found

    def assign_flag(self,finaldf, flag_dic):
        if not flag_dic:  # Check if dic is empty
            return finaldf
        # Apply the helper function row-wise
        finaldf = finaldf.apply(lambda row: self.assign_flag_helper(row, flag_dic), axis=1)
        return finaldf

    def assign_flag_helper(self, row, flag_dic):
        myevent = str(row['events']).strip().lower()
        for flag_key, flag_condition in flag_dic.items():
            row[flag_key] = 0
            for event_key in flag_condition:
                if str(event_key).strip().lower() in myevent:
                    row[flag_key] = 1
                    break      
        if row['IND_Tier1']==0 and row['IND_Tier2']==0 and row['IND_Tier3']==0:
            row['IND_Tier4']=1
        else:
            row['IND_Tier4']=0        
        return row

    def save_sheet(self,sheet,name='combined.csv'):
        sheet.to_csv(name,index=False)
