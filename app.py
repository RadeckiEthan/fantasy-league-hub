from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preach')
def preach():
    # Season start date (Week 1) - Tuesdays
    season_start = datetime(2025, 9, 2)  # September 2nd, 2025 (Tuesday)
    today = datetime.now()
    
    # Create list of weeks with calculated dates
    weeks = []
    for week_num in [11, 12]:  # Added Week 12
        week_date = season_start + timedelta(weeks=week_num - 1)
        days_ago = (today - week_date).days
        
        # Format the "time ago" string
        if days_ago == 0:
            time_ago = "today"
        elif days_ago == 1:
            time_ago = "1 day ago"
        elif days_ago < 7:
            time_ago = f"{days_ago} days ago"
        elif days_ago < 14:
            time_ago = "1 week ago"
        elif days_ago < 30:
            time_ago = f"{days_ago // 7} weeks ago"
        elif days_ago < 60:
            time_ago = "1 month ago"
        else:
            time_ago = f"{days_ago // 30} months ago"
        
        weeks.append({
            'number': week_num,
            'time_ago': time_ago
        })
    
    # Reverse so newest is first
    weeks.reverse()
    
    return render_template('preach_hub.html', weeks=weeks)

@app.route('/preach/week12')
def preach_week12():
    return render_template('preach_week12.html')

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

@app.route('/preach/draft-analysis')
def preach_draft_analysis():
    import pandas as pd
    import numpy as np
    
    # Load data
    df = pd.read_csv('static/data/preach_manager_stats.csv', encoding='latin-1')
    
    # Exclude 15-team season (Draft_Slot 15)
    df = df[df['Draft_Slot'] <= 14].copy()
    
    # Calculate manager career averages (for adjustment)
    manager_avg = df.groupby('Manager')['Dominance_Score'].mean().to_dict()
    df['Manager_Quality'] = df['Manager'].map(manager_avg)
    
    # Draft slot analysis
    draft_stats = []
    for slot in range(1, 15):
        slot_data = df[df['Draft_Slot'] == slot]
        
        if len(slot_data) == 0:
            continue
            
        # Raw performance metrics
        playoff_rate = slot_data['Playoffs'].mean() * 100
        champ_rate = slot_data['Champ_W'].mean() * 100
        avg_pfg = slot_data['PF/G'].mean()
        avg_dominance = slot_data['Dominance_Score'].mean()
        
        # Manager-adjusted performance
        expected_dominance = slot_data['Manager_Quality'].mean()
        actual_dominance = avg_dominance
        overperformance = actual_dominance - expected_dominance
        
        # Manager distribution
        manager_counts = slot_data['Manager'].value_counts().to_dict()
        
        draft_stats.append({
            'slot': slot,
            'seasons': len(slot_data),
            'playoff_rate': round(playoff_rate, 1),
            'champ_rate': round(champ_rate, 1),
            'avg_pfg': round(avg_pfg, 2),
            'avg_dominance': round(avg_dominance, 2),
            'expected_dominance': round(expected_dominance, 2),
            'overperformance': round(overperformance, 2),
            'manager_counts': manager_counts
        })
    
    # Manager quality rankings
    manager_quality = sorted(
        [(mgr, round(score, 2)) for mgr, score in manager_avg.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return render_template('preach_draft_analysis.html', 
                         draft_stats=draft_stats,
                         manager_quality=manager_quality)

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
        
        # Clamp values to valid RGB range (0-255)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        
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
    
    # Create Dominance Chart (Dominance_Score)
    # Function to get color based on z-score (positive = green/dominant, negative = red/weak)
    def get_dominance_color(zscore):
        if zscore >= 0:
            # Positive z-score (dominant - high PF/G) - scale to green
            # Cap at z-score of 2 for color scaling
            normalized = min(zscore / 2, 1)
            r = int((1 - normalized) * 255)
            g = 255
            b = int((1 - normalized) * 255)
        else:
            # Negative z-score (weak - low PF/G) - scale to red
            # Cap at z-score of -2 for color scaling
            normalized = min(abs(zscore) / 2, 1)
            r = 255
            g = int((1 - normalized) * 255)
            b = int((1 - normalized) * 255)
        return f'rgb({r}, {g}, {b})'
    
    dominance_colors = [get_dominance_color(z) for z in chart_data['Dominance_Score']]
    
    dominance_chart = go.Figure()
    
    # Add z-score bars
    dominance_chart.add_trace(go.Bar(
        x=chart_data['Year'],
        y=chart_data['Dominance_Score'],
        name='Dominance Score',
        marker=dict(
            color=dominance_colors,
            line=dict(color='black', width=1)
        ),
        hovertemplate='Year: %{x}<br>Dominance Score: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line (league average)
    dominance_chart.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="black", 
        line_width=2,
        annotation_text="League Average (0)",
        annotation_position="right"
    )
    
    dominance_chart.update_layout(
        title='Season Dominance Score (Based on PF/G Z-Score)',
        xaxis_title='Year',
        yaxis_title='Dominance Score (Z-Score)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14),
        hovermode='x unified',
        showlegend=False
    )
    
    dominance_chart_html = pio.to_html(dominance_chart, full_html=False, include_plotlyjs='cdn')
    
    # Calculate best and worst matchups from matchup_data.csv
    # Calculate best and worst matchups from matchup_data.csv
    try:
        matchup_df = pd.read_csv('static/data/matchup_data.csv', encoding='latin-1')
        
        # Filter for this manager and include regular season + playoff games
        manager_matchups = matchup_df[
            (matchup_df['Team_Name'] == manager_name) &
            ((matchup_df['Week'].str.contains(r'^Week \d+', na=False, regex=True)) | 
             ((matchup_df['Week'].str.contains(r'^Playoff Round', na=False, regex=True)) & 
              (matchup_df['Is_Playoff'] == 'Yes')))
        ].copy()
        
        if len(manager_matchups) == 0:
            raise ValueError("No matchups found")
        
        # Calculate head-to-head records
        h2h = manager_matchups.groupby('Opponent_Name').agg({
            'Outcome': lambda x: [(x == 'Win').sum(), (x == 'Loss').sum(), len(x)]
        }).reset_index()
        
        h2h['Wins'] = h2h['Outcome'].apply(lambda x: x[0])
        h2h['Losses'] = h2h['Outcome'].apply(lambda x: x[1])
        h2h['Games'] = h2h['Outcome'].apply(lambda x: x[2])
        h2h['Win_Pct'] = h2h['Wins'] / h2h['Games']
        
        # Exclude William Serafin and Thomas Sullivan
        h2h = h2h[~h2h['Opponent_Name'].isin(['William Serafin', 'Thomas Sullivan'])]
        
        if len(h2h) == 0:
            raise ValueError("No valid opponents found")
        
        # Get best matchup (highest win %)
        best_matchup = h2h.loc[h2h['Win_Pct'].idxmax()]
        worst_matchup = h2h.loc[h2h['Win_Pct'].idxmin()]
        
        matchup_stats = {
            'best_opponent': best_matchup['Opponent_Name'],
            'best_record': f"{int(best_matchup['Wins'])}-{int(best_matchup['Losses'])}",
            'best_win_pct': f"({round(best_matchup['Win_Pct'] * 100, 1)}%)",
            'worst_opponent': worst_matchup['Opponent_Name'],
            'worst_record': f"{int(worst_matchup['Wins'])}-{int(worst_matchup['Losses'])}",
            'worst_win_pct': f"({round(worst_matchup['Win_Pct'] * 100, 1)}%)"
        }
    except Exception as e:
        print(f"Error loading matchup data: {e}")
        matchup_stats = {
            'best_opponent': 'N/A',
            'best_record': 'N/A',
            'best_win_pct': '',
            'worst_opponent': 'N/A',
            'worst_record': 'N/A',
            'worst_win_pct': ''
        }
    
    return render_template('preach_manager_detail.html', 
                         manager_name=manager_name, 
                         seasons=seasons,
                         career_summary=career_summary,
                         combined_chart=combined_chart_html,
                         luck_chart=luck_chart_html,
                         dominance_chart=dominance_chart_html,
                         matchup_stats=matchup_stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)