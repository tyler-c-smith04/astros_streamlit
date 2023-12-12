import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

# Function to create an interactive plot for a given season with custom bar colors
def create_season_plot(astros, season):
    astros_season = astros[astros['Season'] == season]
    astros_season['GB'] = astros_season['GB'].astype(str)
    astros_season['GB_Clean'] = astros_season['GB'].replace('Tied', '0').str.replace(r'\+', '', regex=True)
    astros_season['GB_Clean'] = pd.to_numeric(astros_season['GB_Clean'], errors='coerce')
    astros_season['GB_Clean'] = astros_season['GB_Clean'] * astros_season['GB'].str.contains(r'\+').map({True: 1, False: -1})

    # Use a custom function to determine the color based on the value of 'GB_Clean'
    astros_season['color'] = astros_season['GB_Clean'].apply(lambda x: 'green' if x >= 0 else 'red')

    fig = px.bar(
        astros_season,
        x='Gm#',
        y='GB_Clean',
        color='color',
        color_discrete_map={'green': 'green', 'red': 'red'},  # Custom color map
        labels={'GB_Clean': 'Games Behind (-) / Lead (+)', 'Gm#': 'Games'},
        title=f'Houston Astros {season} Division Lead/Trail'
    )

    # Update layout for larger fonts
    fig.update_layout(
        title_font_size=18,
        xaxis_title='Games',
        yaxis_title='Games Behind (-) / Lead (+)',
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        legend_title_font_size=14,
        showlegend=False  # Hide the legend if you don't need it
    )

    # Add custom data for the tooltip
    fig.update_traces(
        hovertemplate="<b>Game:</b> %{x}<br><b>GB:</b> %{y}",
        marker_line_color='rgb(0,0,0)',  # Add borders to the bars to distinguish them better
        marker_line_width=1.5
    )

    return fig

# Streamlit app
def main():
    st.title("Houston Astros Divisional Leads and Deficits")
    st.write('Select a season to explore how many games the Astros have led or trailed in their division by season! Hover over the chart to see how many games the Astros were leading or trailing their division at any given moment during the season.')

    # Load your data here
    astros = pd.read_csv('astros.csv')

    seasons = range(2017, 2024)
    selected_season = st.selectbox("Select a Season", seasons)

    fig = create_season_plot(astros, selected_season)
    st.plotly_chart(fig, use_container_width=True)  # Use the full width of the container

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import plotly.express as px

# Function to create an interactive histogram for a given season and run type
def create_runs_histogram(astros, season, run_type):
    astros_season = astros[astros['Season'] == season]
    
    # Define the column name and color based on the run type
    column = 'R' if run_type == 'Runs Scored' else 'RA'
    color = '#002D62' if run_type == 'Runs Scored' else '#EB6E1F'

    # Create the histogram using Plotly
    fig = px.histogram(
        astros_season, 
        x=column, 
        nbins=24, 
        title=f'Distribution of {run_type} by the Astros in {season}',
        labels={column: 'Runs' if run_type == 'Runs Scored' else 'Runs Allowed'},
        color_discrete_sequence=[color]
    )

    # Update the layout for larger fonts and add tooltips
    fig.update_layout(
        title_font_size=18,
        xaxis_title='Runs' if run_type == 'Runs Scored' else 'Runs Allowed',
        yaxis_title='Frequency',
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12
    )

    fig.update_traces(
        hoverinfo='x+y',  # Show x and y values in the tooltip
        hovertemplate='<b>Count:</b> %{y}<br><b>Runs:</b> %{x}<extra></extra>',
        marker_line_color='rgb(0,0,0)',  # Add borders to the bars
        marker_line_width=1.5
    )

    return fig

# Streamlit app
def main():
    st.title("Astros Runs Distribution")
    st.write('Select a season and whether you want to view the distribution for runs scored by the Astros or runs allowed by the Astros.')

    # Load your data here
    astros = pd.read_csv('astros.csv')

    seasons = astros['Season'].unique()
    selected_season = st.selectbox("Select a Season", seasons)

    # Options for run type
    run_type_option = st.selectbox("Select Run Type", ['Runs Scored', 'Runs Allowed'])
    
    # Create and display the histogram
    fig = create_runs_histogram(astros, selected_season, run_type_option)
    st.plotly_chart(fig, use_container_width=True)  # Use the full width of the container

if __name__ == "__main__":
    main()


# Streamlit app

def main():
    st.title("Record Against Opponents")
    st.write("The Astros' success between 2017 and 2023 has come at the expense of many around MLB. Search for your favorite team to see how they have done against the Astros!")

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

    #plt.title('Houston Astros Wins and Losses by Season')
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
    st.write("The Astros have won more than 90 games each season between 2017 and 2023, with the exception of a shortened season in 2020 where each team only played 60 games. For each season that you select, you will see the Astros' record along with their longest winning and losing streaks for that season.")

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
    # st.write("Houston Astros Wins and Losses by Season")
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
import plotly.express as px

def ensure_w_l_columns(df):
    if 'W' not in df.columns:
        df['W'] = 0
    if 'L' not in df.columns:
        df['L'] = 0
    return df

def plot_win_percentage_by_team(team_win_loss, division):
    fig = px.bar(
        team_win_loss,
        x='Opp',
        y='Win_Percentage',
        title=f'Win Percentage of Houston Astros against {division} (2017-2023)',
        labels={'Opp': 'Teams', 'Win_Percentage': 'Win Percentage'},
        color_discrete_sequence=['#002D62']  # Astros' navy blue color
    )
    
    fig.update_traces(hovertemplate='<b>Team:</b> %{x}<br><b>Win Percentage:</b> %{y:.3f}')
    fig.update_layout(
        xaxis_title='Teams',
        yaxis_title='Win Percentage',
        yaxis_tickformat='.3f',
        showlegend=False
    )

    return fig

def main():
    st.title('Astros Success Against Each Division')
    st.write("Explore the Astros' success against the different MLB divisions. The Astros face the teams in the American League every year and only a select number of teams from the National League each season. For that reason, sample sizes against the National League divisions will be smaller.")
    astros = pd.read_csv('astros.csv')

    season = st.selectbox('Select Season', sorted(astros['Season'].unique()))
    division = st.selectbox('Select Division', ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West'])

    # Ensure this mapping is correct and matches the 'Opp' column values in your CSV
    divisions_to_teams = {
        'AL East': ['NYY', 'BAL', 'TOR', 'BOS', 'TB'],
        'AL Central': ['CLE', 'DET', 'CHW', 'KCR', 'MIN'],
        'AL West': ['HOU', 'LAA', 'OAK', 'SEA', 'TEX'],
        'NL East': ['NYM', 'ATL', 'MIA', 'PHI', 'WSN'],
        'NL Central': ['CHC', 'MIL', 'CIN', 'PIT', 'STL'],
        'NL West': ['LAD', 'SDP', 'SFG', 'COL', 'ARI']
    }

    # Filter for games against division teams across all years
    division_data_all_years = astros[astros['Opp'].isin(divisions_to_teams[division])]
    win_loss_all_years = division_data_all_years.groupby(['Opp', 'W/L'])['W/L'].size().unstack(fill_value=0)
    win_loss_all_years = ensure_w_l_columns(win_loss_all_years)
    win_loss_all_years['Win_Percentage'] = win_loss_all_years['W'] / (win_loss_all_years['W'] + win_loss_all_years['L'])
    win_loss_all_years.reset_index(inplace=True)
    win_loss_all_years['Win_Percentage_Text'] = win_loss_all_years['Win_Percentage'].apply(lambda x: f'{x:.3f}')

    # Filter for the selected season
    season_data = astros[(astros['Season'] == season) & (astros['Opp'].isin(divisions_to_teams[division]))]
    win_loss_season = season_data.groupby(['Opp', 'W/L'])['W/L'].size().unstack(fill_value=0)
    win_loss_season = ensure_w_l_columns(win_loss_season)
    win_loss_season.reset_index(inplace=True)

    # Get overall win/loss record for the selected season
    overall_wins = win_loss_season['W'].sum()
    overall_losses = win_loss_season['L'].sum()
    overall_record_df = pd.DataFrame({'Wins': [overall_wins], 'Losses': [overall_losses]})

    # Layout for chart and tables
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_win_percentage_by_team(win_loss_all_years, division)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.caption(f"Season {season} record against teams in the {division}")
        st.table(win_loss_season[['Opp', 'W', 'L']].set_index('Opp'))
        
        st.caption(f"Overall record against {division} in {season}:")
        st.table(overall_record_df)

if __name__ == "__main__":
    main()








