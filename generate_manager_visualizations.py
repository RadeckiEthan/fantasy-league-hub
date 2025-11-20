import pandas as pd
import plotly.graph_objects as go
import os

# Read the data
df = pd.read_csv('static/data/matchup_data.csv', encoding='latin-1')

# Filter to include regular season games AND actual playoff games
filtered_df = df[
    (df['Week'].str.contains(r'^Week \d+', na=False, regex=True)) | 
    ((df['Week'].str.contains(r'^Playoff Round', na=False, regex=True)) & (df['Is_Playoff'] == 'Yes'))
].copy()

# Get global min and max for histogram (round to nearest 5)
global_min = (filtered_df['Team_Score'].min() // 5) * 5
global_max = ((filtered_df['Team_Score'].max() // 5) + 1) * 5

def create_seasonal_trends(manager_name, data):
    """Create quarterly performance trends chart"""
    
    # Filter for regular season only for quarterly analysis
    manager_data = data[
        (data['Team_Name'] == manager_name) & 
        (data['Week'].str.contains(r'^Week \d+', na=False, regex=True))
    ].copy()
    
    # Extract week number
    manager_data['Week_Num'] = manager_data['Week'].str.extract(r'(\d+)').astype(int)
    
    # Create quarters
    def assign_quarter(week_num):
        if 1 <= week_num <= 3:
            return 'Weeks 1-3'
        elif 4 <= week_num <= 6:
            return 'Weeks 4-6'
        elif 7 <= week_num <= 9:
            return 'Weeks 7-9'
        elif 10 <= week_num <= 13:
            return 'Weeks 10-13'
        else:
            return 'Other'
    
    manager_data['Quarter'] = manager_data['Week_Num'].apply(assign_quarter)
    manager_data = manager_data[manager_data['Quarter'] != 'Other']
    
    # Calculate quarterly stats
    quarterly_stats = manager_data.groupby(['Season_Year', 'Quarter']).agg({
        'Outcome': lambda x: (x == 'Win').sum(),
        'Team_Score': 'mean'
    }).reset_index()
    
    quarterly_stats.columns = ['Season_Year', 'Quarter', 'Wins', 'Avg_Points']
    
    # Calculate losses
    games_per_quarter = manager_data.groupby(['Season_Year', 'Quarter']).size().reset_index(name='Games')
    quarterly_stats = quarterly_stats.merge(games_per_quarter, on=['Season_Year', 'Quarter'])
    quarterly_stats['Losses'] = quarterly_stats['Games'] - quarterly_stats['Wins']
    quarterly_stats['Record'] = quarterly_stats['Wins'].astype(str) + '-' + quarterly_stats['Losses'].astype(str)
    
    # Create plot
    fig = go.Figure()
    
    # Order quarters correctly
    quarter_order = ['Weeks 1-3', 'Weeks 4-6', 'Weeks 7-9', 'Weeks 10-13']
    
    for season in sorted(quarterly_stats['Season_Year'].unique()):
        season_data = quarterly_stats[quarterly_stats['Season_Year'] == season]
        season_data['Quarter'] = pd.Categorical(season_data['Quarter'], categories=quarter_order, ordered=True)
        season_data = season_data.sort_values('Quarter')
        
        fig.add_trace(go.Scatter(
            x=season_data['Quarter'],
            y=season_data['Avg_Points'],
            mode='lines+markers',
            name=f'{season}',
            text=[f"Season {season}<br>{q}<br>Record: {r}<br>Avg Points: {round(p, 2)}" 
                  for q, r, p in zip(season_data['Quarter'], season_data['Record'], season_data['Avg_Points'])],
            hoverinfo='text',
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title=f"{manager_name} - Quarterly Performance Trends",
        xaxis_title="Quarter of Season",
        yaxis_title="Average Points Scored",
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14)
    )
    
    return fig

def create_h2h_heatmap(manager_name, data):
    """Create head-to-head record heatmap"""
    
    # Calculate head-to-head records
    h2h_data = data[data['Team_Name'] == manager_name].copy()
    
    h2h_records = h2h_data.groupby('Opponent_Name').agg({
        'Outcome': lambda x: [(x == 'Win').sum(), (x == 'Loss').sum(), len(x)]
    }).reset_index()
    
    h2h_records['Wins'] = h2h_records['Outcome'].apply(lambda x: x[0])
    h2h_records['Losses'] = h2h_records['Outcome'].apply(lambda x: x[1])
    h2h_records['Games'] = h2h_records['Outcome'].apply(lambda x: x[2])
    h2h_records['Win_Pct'] = h2h_records['Wins'] / h2h_records['Games']
    h2h_records = h2h_records.sort_values('Win_Pct', ascending=False)
    
    # Build custom hover text
    hover_text = [[f"vs {opp}<br>Record: {w}-{l}<br>Win %: {round(wp*100, 1)}%" 
                   for opp, w, l, wp in zip(h2h_records['Opponent_Name'], 
                                           h2h_records['Wins'], 
                                           h2h_records['Losses'], 
                                           h2h_records['Win_Pct'])]]
    
    # Create cleaner heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[h2h_records['Win_Pct'].values],
        x=h2h_records['Opponent_Name'].values,
        y=['Win %'],
        colorscale=[[0, 'rgb(211, 47, 47)'], [0.5, 'rgb(255, 243, 224)'], [1, 'rgb(56, 142, 60)']],
        showscale=True,
        colorbar=dict(
            title="Win %",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            tickformat=".0%",
            len=0.7
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{manager_name} - Head-to-Head Win Percentage",
        xaxis_title="Opponent",
        xaxis={'tickangle': -45, 'tickfont': {'size': 11}},
        yaxis_title="",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        height=280,
        margin=dict(l=50, r=150, t=80, b=130)
    )
    
    return fig

def create_scoring_histogram(manager_name, data, min_score, max_score):
    """Create scoring distribution histogram"""
    
    manager_scores = data[data['Team_Name'] == manager_name]['Team_Score'].values
    
    fig = go.Figure(data=[go.Histogram(
        x=manager_scores,
        xbins=dict(start=min_score, end=max_score, size=5),
        marker_color='#6c5a6c',
        marker_line_color='#4a3a4a',
        marker_line_width=1.5
    )])
    
    fig.update_layout(
        title=f"{manager_name} - Scoring Distribution (5-Point Bins)",
        xaxis_title="Points Scored",
        yaxis_title="Frequency",
        xaxis=dict(range=[min_score, max_score]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        bargap=0.05
    )
    
    return fig

def generate_all_visualizations():
    """Generate visualizations for all managers"""
    
    # Create output directory if it doesn't exist
    output_dir = 'static/manager_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique managers
    managers = filtered_df['Team_Name'].unique()
    
    for manager in managers:
        print(f"Generating charts for {manager}...")
        
        # Create charts
        trends_fig = create_seasonal_trends(manager, filtered_df)
        h2h_fig = create_h2h_heatmap(manager, filtered_df)
        histogram_fig = create_scoring_histogram(manager, filtered_df, global_min, global_max)
        
        # Save as HTML
        safe_name = manager.replace(' ', '_').replace('.', '')
        trends_fig.write_html(f'{output_dir}/{safe_name}_seasonal_trends.html')
        h2h_fig.write_html(f'{output_dir}/{safe_name}_h2h_records.html')
        histogram_fig.write_html(f'{output_dir}/{safe_name}_scoring_dist.html')
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_all_visualizations()