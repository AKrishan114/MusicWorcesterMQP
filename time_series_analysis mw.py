import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

file_path = r"C:\Users\akris\OneDrive\Desktop\Music Worcester Work\anon_DataMerge20250210 - anon_DataMerge.csv"
data = pd.read_csv(file_path)

data['EventDate'] = pd.to_datetime(data['EventDate'])
data['CreatedDate'] = pd.to_datetime(data['CreatedDate'])

data = data[(data['CreatedDate'] <= data['EventDate']) & 
            (data['EventDate'].dt.year <= 2026) & 
            (data['Subscriber'] != True) & 
            (data['ChorusMember'] != True) & 
            (data['Student'] != True) & 
            (data['EventType'] != 'Virtual')]

# df = data.copy()
# df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')

# # Drop rows where EventDate is missing
# df = df.dropna(subset=['EventDate'])


# df['OrderSource'] = df['OrderSource'].replace({'Facebook': 'Social Media (Facebook, Instagram, etc.)'})
# df = df[df['OrderSource'] != "I have attended Music Worcester presentations before"]

# df['OrderSource'] = df['OrderSource'].replace({
#     'Email from Music Worcester': 'Email',
#     'Mailing (postcard/brochure) from Music Worcester': 'Mailing',
#     'Heard from a friend': 'Friend',
#     'Social Media (Facebook, Instagram, etc.)': 'Social Media',
#     'Newspaper/Magazine ad': 'Newspaper/Magazine',
#     'Online ad': 'Online',
#     'Hanover Theatre Marketing': 'Hanover Theatre',
#     'Music Worcester\'s website ': 'MW\'s website',
#     'Email or mailing from another organization': 'Other Org'

# })

# # Revenue by OrderSource
# revenue_by_ordersource = df.groupby('OrderSource')['TicketTotal'].sum().sort_values(ascending=False)

# # OrderSource vs. Ticket Quantity
# quantity_by_ordersource = df.groupby('OrderSource')['Quantity'].mean()

# # Count of instances per OrderSource
# count_by_ordersource = df['OrderSource'].value_counts()

# # Combine revenue, quantity, and count into a single DataFrame
# ordersource_summary = pd.DataFrame({
#     'Total Revenue': revenue_by_ordersource,
#     'Average Ticket Quantity': quantity_by_ordersource,
#     'Count of Instances': count_by_ordersource
# }).fillna(0)

# # Sort data by total revenue
# ordersource_summary = ordersource_summary.sort_values(by='Total Revenue', ascending=False)

# # Display results
# print("\nOrderSource Summary:")
# print(ordersource_summary)

# # Visualization for Total Revenue only
# plt.figure(figsize=(10, 6))
# bars = plt.barh(ordersource_summary.index, ordersource_summary['Total Revenue'], color='royalblue')

# # Add labels to the right side of the bars
# for bar in bars:
#     plt.text(bar.get_width() + 5000,  # Adjust spacing
#              bar.get_y() + bar.get_height() / 2,
#              f'${bar.get_width():,.0f}',  # Format as currency
#              va='center', ha='left', fontsize=10)

# plt.xlabel("Revenue ($)")
# plt.ylabel("Order Source")
# plt.title("Total Revenue by Order Source")
# plt.gca().invert_yaxis()  # Invert y-axis so the highest value is at the top
# plt.show()



reference_date = data[['EventDate', 'CreatedDate']].min().min()

def WeekDifference(data, reference_date):
    data = data.copy()
    data['EventWeek'] = ((data['EventDate'] - reference_date).dt.days // 7) + 1
    data['CreatedWeek'] = ((data['CreatedDate'] - reference_date).dt.days // 7) + 1
    data['WeekDifference'] = data['EventWeek'] - data['CreatedWeek']
    return data

def MonthDifference(data, reference_date):
    data = data.copy()
    data['EventMonth'] = ((data['EventDate'].dt.year - reference_date.year) * 12 +
                          data['EventDate'].dt.month - reference_date.month)
    data['CreatedMonth'] = ((data['CreatedDate'].dt.year - reference_date.year) * 12 +
                            data['CreatedDate'].dt.month - reference_date.month)
    data['MonthDifference'] = data['EventMonth'] - data['CreatedMonth']
    return data


data = WeekDifference(data, reference_date)
data = MonthDifference(data, reference_date)

pre_pandemic_cutoff = pd.Timestamp("2020-03-01")
post_pandemic_start = pd.Timestamp("2021-03-01")

data_pre_pandemic = data[data['EventDate'] < pre_pandemic_cutoff]
data_post_pandemic = data[data['EventDate'] > post_pandemic_start]

unique_event_classes = data['EventClass'].dropna().unique()
color_map = {event: plt.get_cmap("tab10")(i) for i, event in enumerate(unique_event_classes)}

def survival_km_weekdifference(data, column, ax, title, color_map):
    data_filtered = data[(data['WeekDifference'] < 100) & (data[column].notna())].copy()
    data_filtered['Event'] = 1  # Necessary for Kaplan-Meier fitting
    kmf = KaplanMeierFitter()

    for item in data_filtered[column].unique():
        subset = data_filtered[data_filtered[column] == item]
        kmf.fit(durations=subset['WeekDifference'], event_observed=subset['Event'], label=str(item))
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color_map[item])

    ax.set_title(title)
    ax.set_xlabel('Weeks Before Event')
    ax.set_ylabel('Survival Probability')
    ax.set_ylim(0, 1)  
    ax.set_xlim(0, 50)  
    ax.legend()
    ax.grid(True)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)  # Add median survival reference line

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

survival_km_weekdifference(data_pre_pandemic, 'EventClass', axes[0], 'Pre-Pandemic (Before Mar 2020)', color_map)
survival_km_weekdifference(data_post_pandemic, 'EventClass', axes[1], 'Post-Pandemic (After Mar 2021)', color_map)

plt.tight_layout()
plt.show()


# def MonthLabels():
#     first_of_month = reference_date.replace(day=1)
#     while first_of_month < data['EventDate'].max():
#         week_diff = (first_of_month - reference_date).days // 7 + 1
#         plt.axvline(x=week_diff, color='grey', linestyle='--', linewidth=0.7)

#         # plt.text(week_diff, 1.02, first_of_month.strftime('%Y-%m-%d'), 
#         #             rotation=90, verticalalignment='bottom', fontsize=8, color='grey')

#         first_of_month += pd.DateOffset(months=1)


# #Basic Time-Series Graph
# def CWeekvsEWeek(data, column):

#     created_events = data.groupby('CreatedWeek').size()
#     weekly_events = data.groupby('EventWeek').size()


#     plt.figure(figsize=(12, 6))
#     created_events.plot(label="Created Events", linestyle='--', color='blue')
#     weekly_events.plot(label="Weekly Events", color='orange')
#     MonthLabels()

#     plt.title("Total Number of Events and Ticket Purchases Per Week")
#     plt.xlabel("Weeks")
#     plt.ylabel("Number of Instances")
#     plt.legend()  
#     plt.grid(True)
#     plt.tight_layout()  
#     plt.show()

#     unique_items = data[column].dropna().unique()
    
#     for item in unique_items:
#         item_data = data[data[column] == item]

#         created_events = item_data.groupby('CreatedWeek').size()
#         # plt.figure(figsize=(12, 6))
#         # created_events.plot(label=f"Created Events - {item}", linestyle='--', color='blue')
#         # plot_acf(created_events, lags=156, title=f"ACF for Created Events - {item}")
#         # plt.xlabel("Lags (Weeks)")
#         # plt.ylabel("Autocorrelation")
#         # plt.grid(True)
#         # plt.tight_layout()
#         # plt.show()

        
#         decomposition = seasonal_decompose(created_events, model='additive', period=152)
        
    
#         plt.figure(figsize=(14, 8))
#         plt.suptitle(f'Additive Decomposition for {item}', fontsize=16)
        
#         plt.subplot(4, 1, 1)
#         plt.plot(decomposition.observed, label='Observed')
#         plt.legend(loc='upper left')
        
#         plt.subplot(4, 1, 2)
#         plt.plot(decomposition.trend, label='Trend', color='orange')
#         plt.legend(loc='upper left')
        
#         plt.subplot(4, 1, 3)
#         plt.plot(decomposition.seasonal, label='Seasonal', color='green')
#         plt.legend(loc='upper left')
        
#         plt.subplot(4, 1, 4)
#         plt.plot(decomposition.resid, label='Residual', color='red')
#         plt.legend(loc='upper left')
#         plt.xlabel("Weeks")
        
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.show()
        
# CWeekvsEWeek(data, 'EventGenre')


# #Ticket Purchase Instances by Genre Over Weeks
# genre_weekly = data.groupby(['CreatedWeek', 'EventGenre']).size().unstack(fill_value=0)
# genre_weekly = genre_weekly.tail(52)
# genre_weekly.plot(kind='bar', stacked=True, figsize=(14, 8), cmap=cm.get_cmap("Set3"))
# first_of_month = reference_date.replace(day=1)
# MonthLabels()

# plt.title("Ticket Purchase Instances by Genre Over Weeks")
# plt.xlabel("Weeks")
# plt.ylabel("Number of Ticket Purchase Instances")
# plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='y')  
# plt.tight_layout()  
# plt.show()


# # def add_day_of_week(data):
# #     data['CreatedDate'] = pd.to_datetime(data['CreatedDate'])
# #     data['DayOfWeek'] = data['CreatedDate'].dt.day_name()
# #     plt.figure(figsize=(10, 6))
# #     sns.countplot(data=data, x='DayOfWeek', order=[
# #         'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# #     plt.title('Distribution of Days of the Week')
# #     plt.xlabel('Day of the Week')
# #     plt.ylabel('Count')
# #     plt.xticks(rotation=45)
# #     plt.grid(axis='y')
# #     plt.tight_layout()
# #     plt.show()
# #     print(data)
# # add_day_of_week(data)
