import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Function to create plots for a given season
def create_season_plot(astros, season):
    astros_season = astros[astros['Season'] == season]
    astros_season['GB'] = astros_season['GB'].astype(str)
    astros_season['GB_Clean'] = astros_season['GB'].replace('Tied', '0').str.replace(r'\+', '', regex=True)
    astros_season['GB_Clean'] = pd.to_numeric(astros_season['GB_Clean'], errors='coerce')
    astros_season['GB_Clean'] = astros_season['GB_Clean'] * astros_season['GB'].str.contains(r'\+').map({True: 1, False: -1})
    astros_season['color'] = ['green' if x >= 0 else 'red' for x in astros_season['GB_Clean']]
    
    max_lead = astros_season['GB_Clean'].max()
    max_deficit = astros_season['GB_Clean'].min()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(astros_season['Gm#'], astros_season['GB_Clean'], color=astros_season['color'])
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_xlabel('Games')
    ax.set_ylabel('Games Behind (-) / Lead (+)')
    ax.set_title(f'Houston Astros {season} Division Lead/Trail')
    ax.text(0.5, 0.9, f'Max Lead: {max_lead} games\nMax Deficit: {max_deficit} games', 
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    return fig

# Streamlit app
def main():
    st.title("Houston Astros Divisional Leads and Deficits")

    # Load your data here
    astros = pd.read_csv('astros.csv')


    seasons = range(2017, 2024)
    selected_season = st.selectbox("Select a Season", seasons)

    fig = create_season_plot(astros, selected_season)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

# Function to create a histogram for a given season
def create_runs_histogram(astros, season):
    astros_season = astros[astros['Season'] == season]
    plt.figure(figsize=(14, 8))
    sns.histplot(astros_season['R'], kde=False, color='#002D62', bins=24)  # 24 bins for 0 to 23 runs
    plt.title(f'Distribution of Runs Scored by the Astros in {season}')
    plt.xlabel('Runs Scored')
    plt.ylabel('Frequency')
    plt.xticks(range(24))
    return plt

# Function to create a histogram for a given season and run type
def create_runs_histogram(astros, season, run_type):
    astros_season = astros[astros['Season'] == season]
    plt.figure(figsize=(14, 8))
    sns.histplot(astros_season[run_type], kde=False, color='#002D62', bins=24)  # 24 bins for 0 to 23 runs
    plt.title(f'Distribution of {run_type} by the Astros in {season}')
    plt.xlabel(run_type)
    plt.ylabel('Frequency')
    plt.xticks(range(24))
    return plt

# Function to create a histogram for a given season and run type
def create_runs_histogram(astros, season, run_type):
    astros_season = astros[astros['Season'] == season]
    plt.figure(figsize=(14, 8))
    
    # Ensure that the column name is correct
    column = 'R' if run_type == 'Runs Scored' else 'RA'
    
    sns.histplot(astros_season[column], kde=False, color='#002D62', bins=24)  # 24 bins for 0 to 23 runs
    plt.title(f'Distribution of {run_type} by the Astros in {season}')
    plt.xlabel(run_type)
    plt.ylabel('Frequency')
    plt.xticks(range(24))
    return plt

# Function to create a histogram for a given season and run type
def create_runs_histogram(astros, season, run_type):
    astros_season = astros[astros['Season'] == season]
    plt.figure(figsize=(14, 8))
    
    # Select the color based on the run type
    color = '#002D62' if run_type == 'Runs Scored' else '#EB6E1F'
    column = 'R' if run_type == 'Runs Scored' else 'RA'
    
    sns.histplot(astros_season[column], kde=False, color=color, bins=24)  # 24 bins for 0 to 23 runs
    plt.title(f'Distribution of {run_type} by the Astros in {season}')
    plt.xlabel('Runs' if run_type == 'Runs Scored' else 'Runs Allowed')
    plt.ylabel('Frequency')
    plt.xticks(range(24))
    return plt

# Streamlit app
def main():
    st.title("Astros Runs Distribution")

    # Load your data here
    astros = pd.read_csv('astros.csv')


    seasons = astros['Season'].unique()
    selected_season = st.selectbox("Select a Season", seasons)

    # Options for run type
    run_type_option = st.selectbox("Select Run Type", ['Runs Scored', 'Runs Allowed'])
    
    sns.set_style("whitegrid")
    
    # Create and display the histogram
    plt = create_runs_histogram(astros, selected_season, run_type_option)
    st.pyplot(plt)

if __name__ == "__main__":
    main()

# Streamlit app

def main():
    st.title("Record Against Opponents")

    # Load your data here if it's not already loaded
    astros = pd.read_csv('astros.csv')

    # Dropdown for selecting the opponent, sorted alphabetically
    opponents = sorted(astros['Opp'].unique())
    selected_opponent = st.selectbox("Select an Opponent", opponents)

    # Use groupby and ensure 'W' and 'L' are present
    win_loss_counts = astros.groupby('Opp')['W/L'].value_counts().unstack(fill_value=0)
    win_loss_counts = win_loss_counts[['W', 'L']] if 'W' in win_loss_counts and 'L' in win_loss_counts else win_loss_counts

    # Calculate win percentage
    win_loss_counts['Win_Percentage'] = win_loss_counts['W'] / (win_loss_counts['W'] + win_loss_counts['L'])

    # Display the record against the selected opponent
    if selected_opponent in win_loss_counts.index:
        st.write(f"Record against {selected_opponent} between 2017-2023:")
        st.write(win_loss_counts.loc[selected_opponent])
    else:
        st.error(f"No data available for {selected_opponent}.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot the overall wins and losses over time
def plot_overall_wins_losses(astros):
    astros_season_record_filtered = astros[astros['Season'] != 2020]  # Exclude the 2020 season

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Season', y='Wins', data=astros_season_record_filtered, marker='o', label='Wins', color='#002D62')
    sns.lineplot(x='Season', y='Losses', data=astros_season_record_filtered, marker='o', label='Losses', color='#EB6E1F')

    plt.title('Houston Astros Wins and Losses by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Wins/Losses')

    # Set y-axis to start at 60 and intervals of 5
    y_max = max(astros_season_record_filtered['Wins'].max(), astros_season_record_filtered['Losses'].max())
    plt.ylim(40, y_max)
    plt.yticks(range(40, y_max + 5, 5))

    plt.legend()
    plt.grid(True)
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot the overall wins and losses over time
def plot_overall_wins_losses(astros_season_record_filtered):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Season', y='Wins', data=astros_season_record_filtered, marker='o', label='Wins', color='#002D62')
    sns.lineplot(x='Season', y='Losses', data=astros_season_record_filtered, marker='o', label='Losses', color='#EB6E1F')

    plt.title('Houston Astros Wins and Losses by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Wins/Losses')

    # Set y-axis to start at 40 and intervals of 5
    y_max = max(astros_season_record_filtered['Wins'].max(), astros_season_record_filtered['Losses'].max()) + 5
    plt.ylim(40, y_max)
    plt.yticks(range(40, y_max + 1, 5))

    plt.legend()
    plt.grid(True)
    return plt

# Streamlit app
def main():
    st.title("Astros Regular Season Success")

    # Load your data here if it's not already loaded
    astros = pd.read_csv('astros.csv')

    # Calculate the length of the streaks
    astros['Win Streak'] = astros['Streak'].apply(lambda x: x.count('+'))
    astros['Lose Streak'] = astros['Streak'].apply(lambda x: x.count('-'))

    # Group by 'Season' and find the longest winning and losing streaks
    longest_streaks = astros.groupby('Season').agg({
        'Win Streak': 'max',
        'Lose Streak': 'max'
    })

    # Group by the 'Season' column and calculate Wins, Losses, and Win percentage
    astros_season_record = astros.groupby('Season').agg(
        Wins=('W/L', lambda x: (x == 'W').sum()),  # Count 'W'
        Losses=('W/L', lambda x: (x == 'L').sum())  # Count 'L'
    ).astype(int)  # Convert to integers

    # Calculate win percentage
    astros_season_record['Win_Percentage'] = (astros_season_record['Wins'] / 
                                              (astros_season_record['Wins'] + astros_season_record['Losses'])).round(3)

    # Exclude the 2020 season
    astros_season_record_filtered = astros_season_record[astros_season_record.index != 2020]

    # Plot the overall wins and losses chart
    st.write("Houston Astros Wins and Losses by Season")
    plt = plot_overall_wins_losses(astros_season_record_filtered.reset_index())
    st.pyplot(plt)

    # Dropdown for selecting a season
    seasons = astros_season_record.index.tolist()
    selected_season = st.selectbox("Select a Season", seasons, key='season_select')

    # Layout to display records and streaks side by side
    col1, col2 = st.columns(2)

    with col1:
        # Display the record for the selected season
        if selected_season in astros_season_record.index:
            season_record = astros_season_record.loc[selected_season]
            st.write(f"Wins: {season_record['Wins']}")
            st.write(f"Losses: {season_record['Losses']}")
            st.write(f"Win Percentage: {season_record['Win_Percentage']}")

    with col2:
        # Display the longest streaks for the selected season
        if selected_season in longest_streaks.index:
            season_streaks = longest_streaks.loc[selected_season]
            st.write(f"Longest Winning Streak: {season_streaks['Win Streak']}")
            st.write(f"Longest Losing Streak: {season_streaks['Lose Streak']}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot win percentage by division
def plot_win_percentage_by_division(division_group_sorted):
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Division', data=division_group_sorted, palette=['#002D62', '#EB6E1F'])

    plt.title('Win Percentage by Division')
    plt.xlabel('Win Percentage')
    plt.ylabel('Division')

    # Annotate each bar with the value of the win percentage
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.3f}',
                 ha='left', va='center')
    plt.tight_layout()
    return plt


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Astros Success Against Each Division')

astros = pd.read_csv('astros.csv')


# Streamlit user interface for selecting season and division
season = st.selectbox('Select Season', range(2017, 2024))
division = st.selectbox('Select Division', ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West'])

# Filter data based on selected season
astros_season = astros[astros['Season'] == season]

# Create a DataFrame to count wins and losses
win_loss_counts = astros_season.groupby(['Opp', 'W/L']).size().unstack(fill_value=0).reset_index()

# Division mapping
divisions_to_teams = {
    'AL East': ['NYY', 'BAL', 'TOR', 'BOS', 'TB'],
    'AL Central': ['CLE', 'DET', 'CHW', 'KCR', 'MIN'],
    'AL West': ['HOU', 'LAA', 'OAK', 'SEA', 'TEX'],
    'NL East': ['NYM', 'ATL', 'MIA', 'PHI', 'WSN'],
    'NL Central': ['CHC', 'MIL', 'CIN', 'PIT', 'STL'],
    'NL West': ['LAD', 'SDP', 'SFG', 'COL', 'ARI']
}

# Reverse the mapping
team_to_division_mapping = {team: div for div, teams in divisions_to_teams.items() for team in teams}

# Map teams to their divisions
win_loss_counts['Division'] = win_loss_counts['Opp'].map(team_to_division_mapping)

# Filter data based on selected division
division_data = win_loss_counts[win_loss_counts['Division'] == division]

# Check if there is data to display
if division_data.empty:
    st.write(f"No data available for the Houston Astros against the {division} division in {season}.")
else:
    # Calculate win percentage
    division_data['Win_Percentage'] = division_data['W'] / (division_data['W'] + division_data['L'])

    # Create a horizontal bar plot with Astros colors
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Win_Percentage', y='Opp', data=division_data, palette=['#002D62', '#EB6E1F'])

    # Add title and labels
    plt.title(f'Win Percentage of Houston Astros against {division} in {season}')
    plt.xlabel('Win Percentage')
    plt.ylabel('Teams')

    # Annotate each bar
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width if width > 0.01 else 0.01, p.get_y() + p.get_height() / 2, f'{width:.3f}', ha = 'left', va = 'center')

    # Display the plot
    st.pyplot(plt)



