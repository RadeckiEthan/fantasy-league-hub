from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preach')
def preach():
    return render_template('preach_hub.html')

@app.route('/preach/week11')
def preach_week11():
    return render_template('preach_week11.html')

@app.route('/preach/managers')
def preach_managers():
    # Read the CSV with different encoding
    df = pd.read_csv('static/data/preach_manager_stats.csv', encoding='latin-1')
    # Get unique managers and sort alphabetically
    managers = sorted(df['Manager'].unique().tolist())
    return render_template('preach_managers.html', managers=managers)

@app.route('/preach/leaderboard')
def preach_leaderboard():
    # Read the CSV
    df = pd.read_csv('static/data/preach_manager_stats.csv', encoding='latin-1')
    
    # Exclude William Serafin and Thomas Sullivan
    df = df[~df['Manager'].isin(['William Serafin', 'Thomas Sullivan'])]
    
    # Calculate stats for each manager
    leaderboard_data = []
    
    for manager in df['Manager'].unique():
        manager_data = df[df['Manager'] == manager].copy()
        
        total_wins = int(manager_data['W'].sum())
        total_losses = int(manager_data['L'].sum())
        win_pct = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        total_playoffs = int(manager_data['Playoffs'].sum())
        total_championships = int(manager_data['Champ_W'].sum())
        career_pfg = manager_data['PF/G'].mean()
        career_luck = manager_data['LR_zscore'].mean()
        
        # Find best and worst records
        manager_data['Win_Pct'] = manager_data['W'] / (manager_data['W'] + manager_data['L'])
        best_season = manager_data.loc[manager_data['Win_Pct'].idxmax()]
        worst_season = manager_data.loc[manager_data['Win_Pct'].idxmin()]
        
        best_record = f"{int(best_season['W'])}-{int(best_season['L'])} ({int(best_season['Year'])})"
        worst_record = f"{int(worst_season['W'])}-{int(worst_season['L'])} ({int(worst_season['Year'])})"
        
        leaderboard_data.append({
            'Manager': manager,
            'Total_Wins': total_wins,
            'Total_Losses': total_losses,
            'Win_Pct': round(win_pct, 3),
            'Total_Playoffs': total_playoffs,
            'Total_Championships': total_championships,
            'Career_PFG': round(career_pfg, 1),
            'Career_Luck': round(career_luck, 2),
            'Best_Record': best_record,
            'Worst_Record': worst_record
        })
    
    # Convert to DataFrame for easy sorting (default by Win %)
    leaderboard_df = pd.DataFrame(leaderboard_data)
    leaderboard_df = leaderboard_df.sort_values('Win_Pct', ascending=False)
    
    # Convert back to list of dicts for template
    leaderboard = leaderboard_df.to_dict('records')
    
    return render_template('preach_leaderboard.html', leaderboard=leaderboard)

@app.route('/preach/managers/<manager_name>')
def preach_manager_detail(manager_name):
    # Read the CSV with different encoding
    df = pd.read_csv('static/data/preach_manager_stats.csv', encoding='latin-1')
    # Filter for this manager
    manager_data = df[df['Manager'] == manager_name]
    
    if manager_data.empty:
        return "Manager not found", 404
    
    # Function to get color based on rank (1 = green, 14 = red)
    def get_rank_color(rank, total=14):
        # Normalize rank to 0-1 scale (1 is best, 14 is worst)
        normalized = (rank - 1) / (total - 1)
        # Green to Red gradient
        if normalized <= 0.5:
            # Green to Yellow
            r = int(normalized * 2 * 255)
            g = 255
        else:
            # Yellow to Red
            r = 255
            g = int((1 - normalized) * 2 * 255)
        return f'rgb({r}, {g}, 0)'
    
    # Calculate career summary stats
    total_wins = int(manager_data['W'].sum())
    total_losses = int(manager_data['L'].sum())
    overall_win_pct = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
    total_playoffs = int(manager_data['Playoffs'].sum())
    total_championships = int(manager_data['Champ_W'].sum())
    career_pfg = manager_data['PF/G'].mean()
    
    # Get championship years
    champ_years = manager_data[manager_data['Champ_W'] == 1]['Year'].tolist()
    champ_years_str = ', '.join(str(int(year)) for year in champ_years) if champ_years else ''
    
    # Calculate ranks across all managers for career stats (excluding William Serafin and Thomas Sullivan)
    all_managers = df[~df['Manager'].isin(['William Serafin', 'Thomas Sullivan'])].groupby('Manager').agg({
        'W': 'sum',
        'L': 'sum',
        'Playoffs': 'sum',
        'Champ_W': 'sum',
        'PF/G': 'mean'
    }).reset_index()
    
    all_managers['Win_Pct'] = all_managers['W'] / (all_managers['W'] + all_managers['L'])
    all_managers['Total_Games'] = all_managers['W'] + all_managers['L']
    
    # Calculate ranks (1 is best)
    all_managers['Wins_Rank'] = all_managers['W'].rank(ascending=False, method='min')
    all_managers['Losses_Rank'] = all_managers['L'].rank(ascending=True, method='min')  # Lower losses is better
    all_managers['Win_Pct_Rank'] = all_managers['Win_Pct'].rank(ascending=False, method='min')
    all_managers['Playoffs_Rank'] = all_managers['Playoffs'].rank(ascending=False, method='min')
    all_managers['PFG_Rank'] = all_managers['PF/G'].rank(ascending=False, method='min')
    
    # Get this manager's ranks (if they're in the ranking pool)
    if manager_name in ['William Serafin', 'Thomas Sullivan']:
        # For excluded managers, don't show ranks
        career_summary = {
            'total_wins': total_wins,
            'total_losses': total_losses,
            'overall_win_pct': overall_win_pct,
            'total_playoffs': total_playoffs,
            'total_championships': total_championships,
            'career_pfg': career_pfg,
            'champ_years': champ_years_str,
            'show_ranks': False
        }
    else:
        manager_ranks = all_managers[all_managers['Manager'] == manager_name].iloc[0]
        total_managers = len(all_managers)
        
        career_summary = {
            'total_wins': total_wins,
            'total_losses': total_losses,
            'overall_win_pct': overall_win_pct,
            'total_playoffs': total_playoffs,
            'total_championships': total_championships,
            'career_pfg': career_pfg,
            'champ_years': champ_years_str,
            'wins_rank': int(manager_ranks['Wins_Rank']),
            'losses_rank': int(manager_ranks['Losses_Rank']),
            'win_pct_rank': int(manager_ranks['Win_Pct_Rank']),
            'playoffs_rank': int(manager_ranks['Playoffs_Rank']),
            'pfg_rank': int(manager_ranks['PFG_Rank']),
            'wins_color': get_rank_color(manager_ranks['Wins_Rank'], total_managers),
            'losses_color': get_rank_color(manager_ranks['Losses_Rank'], total_managers),
            'win_pct_color': get_rank_color(manager_ranks['Win_Pct_Rank'], total_managers),
            'playoffs_color': get_rank_color(manager_ranks['Playoffs_Rank'], total_managers),
            'pfg_color': get_rank_color(manager_ranks['PFG_Rank'], total_managers),
            'show_ranks': True
        }
    
    # Convert to list of dictionaries and add colors for season table
    seasons = []
    for _, row in manager_data.iterrows():
        season_dict = row.to_dict()
        season_dict['pf_color'] = get_rank_color(row['PF/G_Rank_within_Year'])
        season_dict['pa_color'] = get_rank_color(row['PA/G_Rank_within_Year'])
        seasons.append(season_dict)
    
    # Create chart with dual y-axes
    # Sort by year for proper line chart
    chart_data = manager_data.sort_values('Year')
    
    # Win Percentage Over Time Chart
    chart_data['Win_Pct'] = chart_data['W'] / (chart_data['W'] + chart_data['L'])
    
    # Calculate league average PF/G for each year
    league_avg_by_year = df.groupby('Year')['PF/G'].mean().reset_index()
    league_avg_by_year.columns = ['Year', 'League_Avg_PFG']
    
    # Merge league average with manager data
    chart_data = chart_data.merge(league_avg_by_year, on='Year', how='left')
    
    # Get color for each bar based on PF/G rank
    bar_colors = [get_rank_color(rank) for rank in chart_data['PF/G_Rank_within_Year']]
    
    combined_chart = go.Figure()
    
    # Add PF/G bars with color coding FIRST (right y-axis) - so they're behind
    combined_chart.add_trace(go.Bar(
        x=chart_data['Year'],
        y=chart_data['PF/G'],
        name='PF/G',
        marker=dict(
            color=bar_colors,
            line=dict(color='black', width=1)
        ),
        opacity=0.8,
        yaxis='y2',
        hovertemplate='Year: %{x}<br>PF/G: %{y:.1f}<br>Rank: %{customdata}<extra></extra>',
        customdata=chart_data['PF/G_Rank_within_Year']
    ))
    
    # Add league average PF/G line (right y-axis)
    combined_chart.add_trace(go.Scatter(
        x=chart_data['Year'],
        y=chart_data['League_Avg_PFG'],
        mode='lines',
        name='League Avg PF/G',
        line=dict(color='rgba(80, 80, 80, 0.8)', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Add Win % line LAST (left y-axis) - so it's on top
    combined_chart.add_trace(go.Scatter(
        x=chart_data['Year'],
        y=chart_data['Win_Pct'],
        mode='lines+markers',
        name='Win %',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10),
        yaxis='y'
    ))
    
    combined_chart.update_layout(
        title='Win Percentage & Points Per Game Over Time',
        xaxis_title='Year',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        hovermode='x unified',
        yaxis=dict(
            title='Win Percentage',
            titlefont=dict(color='#4CAF50'),
            tickfont=dict(color='#4CAF50'),
            range=[0, 1]
        ),
        yaxis2=dict(
            title='PF/G',
            titlefont=dict(color='#9c949c'),
            tickfont=dict(color='#9c949c'),
            overlaying='y',
            side='right'
        ),
        showlegend=False,
        barmode='overlay'
    )
    combined_chart_html = pio.to_html(combined_chart, full_html=False, include_plotlyjs='cdn')
    
    # Create Luck Chart (LR_zscore)
    # Function to get color based on z-score (negative = green/lucky, positive = red/unlucky)
    def get_luck_color(zscore):
        if zscore <= 0:
            # Negative z-score (lucky - low PA/G) - scale to green
            # Cap at z-score of -2 for color scaling
            normalized = min(abs(zscore) / 2, 1)
            r = int((1 - normalized) * 255)
            g = 255
            b = int((1 - normalized) * 255)
        else:
            # Positive z-score (unlucky - high PA/G) - scale to red
            # Cap at z-score of 2 for color scaling
            normalized = min(zscore / 2, 1)
            r = 255
            g = int((1 - normalized) * 255)
            b = int((1 - normalized) * 255)
        return f'rgb({r}, {g}, {b})'
    
    luck_colors = [get_luck_color(z) for z in chart_data['LR_zscore']]
    
    luck_chart = go.Figure()
    
    # Add z-score bars
    luck_chart.add_trace(go.Bar(
        x=chart_data['Year'],
        y=chart_data['LR_zscore'],
        name='Luck Score',
        marker=dict(
            color=luck_colors,
            line=dict(color='black', width=1)
        ),
        hovertemplate='Year: %{x}<br>Luck Score: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line (league average)
    luck_chart.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text="League Average (0)",
        annotation_position="right"
    )
    
    luck_chart.update_layout(
        title='Season Luck Score (Based on PA/G Z-Score)',
        xaxis_title='Year',
        yaxis_title='Luck Score (Z-Score)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        hovermode='x unified',
        showlegend=False
    )
    
    luck_chart_html = pio.to_html(luck_chart, full_html=False, include_plotlyjs='cdn')
    
    return render_template('preach_manager_detail.html', 
                         manager_name=manager_name, 
                         seasons=seasons,
                         career_summary=career_summary,
                         combined_chart=combined_chart_html,
                         luck_chart=luck_chart_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)