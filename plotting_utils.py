import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1 as ff1
import fastf1.plotting as ff1_plot
ff1_plot.setup_mpl(color_scheme='fastf1')
ff1.Cache.enable_cache('cache_dir')

from matplotlib import font_manager
from matplotlib.patches import Patch

font_manager.fontManager.addfont('fonts/Formula1-Regular.ttf')
font_manager.fontManager.addfont('fonts/Formula1-Bold.ttf')
font_manager.fontManager.addfont('fonts/Formula1-Black.ttf')
font_manager.fontManager.addfont('fonts/Formula1-Wide.ttf')
plt.rcParams['font.family'] = 'Formula1'

def plot_fastest_lap_times(session, xlim=None):
    """
    Bar plot for each driver's fastest lap time in the session

    session: FastF1 session object
    xlim: Optional tuple (xmin, xmax) to set x-axis limits
    """
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"
    fastest_lap_times = session.laps.groupby('Driver')['LapTime'].min().sort_values(ascending=False).reset_index()
    fastest_lap_times['LapTime'] = fastest_lap_times['LapTime'].dt.total_seconds()

    colors = [ff1.plotting.get_driver_color(driver, session) for driver in fastest_lap_times['Driver']]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=fastest_lap_times, x='LapTime', y='Driver', hue='Driver', palette=colors)
    plt.title(f"Fastest Lap Times\n{session_name}", fontweight='bold', fontsize=16)
    plt.xlabel('Fastest Lap Time (seconds)')
    if xlim:
        plt.xlim(xlim)
    plt.show()

def plot_race_pace(session, driver1, driver2, color1, color2, regions=None):
    """
    Plots the lap times of two drivers over the laps in the session with pit stops marked.

    session: FastF1 session object
    driver1: Abbreviation of the first driver (e.g., 'ANT')
    driver2: Abbreviation of the second driver (e.g., 'RUS')
    color1: Color for the first driver's line and pit stop markers
    color2: Color for the second driver's line and pit stop markers
    regions: Optional list of tuples (start_lap, end_lap, label, color) to highlight specific periods in the race
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"

    # Retrieve lap data for both drivers, ensuring we only include laps with valid LapTime
    laps1 = session.laps.pick_drivers(driver1).dropna(subset=['LapTime'])
    laps2 = session.laps.pick_drivers(driver2).dropna(subset=['LapTime'])
    total_laps = max(laps1['LapNumber'].max(), laps2['LapNumber'].max())
    pit_times1 = laps1[laps1['PitInTime'].notna()]
    pit_times2 = laps2[laps2['PitInTime'].notna()]
    
    # Plot lap times over laps for both drivers
    ax.plot(laps1['LapNumber'], laps1['LapTime'], linewidth=2, label=driver1, color=color1)
    ax.plot(laps2['LapNumber'], laps2['LapTime'], linewidth=2, label=driver2, color=color2)

    # Plot pit stops as vertical markers
    ax.scatter(pit_times1['LapNumber'], pit_times1['LapTime'], color=color1, marker='v', s=100, zorder=5, label='_nolegend_')
    ax.scatter(pit_times2['LapNumber'], pit_times2['LapTime'], color=color2, marker='v', s=100, zorder=5, label='_nolegend_')

    # Optionally highlight specific regions of the race
    if regions:
        for region in regions:
            ax.axvspan(region[0], region[1], label=region[2], color=region[3], alpha=0.15)

    # More plotting aesthetics
    ax.set_xlabel('Lap Number', fontweight='bold')
    ax.set_ylabel('Lap Time', fontweight='bold')
    ax.set_title(f'Race Pace: {driver1} vs {driver2}\n{session_name}', fontweight='bold')
    ax.legend()

    plt.xlim(0, total_laps + 1)
    plt.show()

def plot_race_pace_boxplot(session, drivers, colors, ylim=None):
    """
    Plots boxplots of lap times for two drivers in the session.
    session: FastF1 session object
    drivers: List of driver abbreviations (e.g., ['ANT', 'RUS'])
    colors: List of colors for the boxplots
    ylim: Optional tuple (ymin, ymax) to set y-axis limits
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"

    # Retrieve lap data for the specified drivers and prepare it for plotting
    lap_data = session.laps.pick_drivers(drivers).copy()
    lap_data['Driver'] = pd.Categorical(lap_data['Driver'], categories=drivers, ordered=True)
    lap_data['LapTimeSeconds'] = lap_data['LapTime'].dt.total_seconds()

    # Create boxplot
    sns.boxplot(data=lap_data, y='LapTimeSeconds', x='Driver', hue='Driver', palette=colors)
    plt.title(f'Race Pace for {", ".join(drivers)}\n{session_name}', fontweight='bold')
    plt.ylabel('Lap Time (seconds)', fontweight='bold')
    if ylim:
        plt.ylim(ylim)
    plt.show()

def plot_lap_time_delta(session, driver1, driver2, color1, color2, ylim=None):
    """
    Plots the lap time delta between two drivers over the laps in the session.
    
    session: FastF1 session object
    driver1: Abbreviation of the first driver (e.g., 'ANT')
    driver2: Abbreviation of the second driver (e.g., 'RUS')
    color1: Color for the first driver's line and pit stop markers
    color2: Color for the second driver's line and pit stop markers
    ylim: Optional tuple (ymin, ymax) to set y-axis limits
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"

    # Retrieve lap data for both drivers and compute the gap between driver1 and driver2
    lap_data = session.laps.pick_drivers([driver1, driver2]).dropna(subset=['LapTime'])
    lap_data['LapTimeSeconds'] = lap_data['LapTime'].dt.total_seconds()
    laps_pivoted = lap_data[['LapNumber', 'Driver', 'LapTimeSeconds']].pivot(index='LapNumber', columns='Driver', values='LapTimeSeconds').reset_index().rename_axis(None, axis=1)
    gap = laps_pivoted[driver2] - laps_pivoted[driver1]
    laps = laps_pivoted['LapNumber']
    
    # Plot the lap time delta over the laps of the race
    ax.plot(laps, gap, color='white', linewidth=1, alpha=0.3)
    ax.fill_between(laps, gap, 0, where=(gap > 0), color=color1, alpha=0.7, label=f"{driver1} faster")
    ax.fill_between(laps, gap, 0, where=(gap < 0), color=color2, alpha=0.7, label=f"{driver2} faster")
    ax.axhline(y=0, color='white', linewidth=0.8, linestyle='--')
        
    # Plotting aesthetics
    ax.set_xlabel('Lap Number', fontweight='bold')
    ax.set_ylabel('Gap (seconds)', fontweight='bold')
    ax.set_title(f'Lap Time Delta: {driver1} vs {driver2}\n{session_name}', fontweight='bold')
    ax.legend()

    if ylim:
        plt.ylim(ylim)
    plt.show()

def format_laptime(td):
    """ 
    Returns a string formatted in the way desired for the telemetry chart

    td: DateTime object
    """
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02.0f}:{seconds:06.3f}"

def fastest_lap_comparison(session, driver1, driver2, color1, color2):
    """ 
    Creates a telemetry chart for driver1's and driver2's fastest laps in the session,
    comparing the lap time delta, speed, throttle, brake, gear, and RPM over the distance of the lap.

    session: FastF1 session object
    driver1: Abbreviation of the first driver (e.g., 'ANT')
    driver2: Abbreviation of the second driver (e.g., 'RUS')
    color1: Color for the first driver's line and pit stop markers
    color2: Color for the second driver's line and pit stop markers
    """
    
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"
    
    fastest_driver1 = session.laps.pick_drivers(driver1).dropna(subset=['LapTime']).pick_fastest()
    fastest_driver2 = session.laps.pick_drivers(driver2).dropna(subset=['LapTime']).pick_fastest()
    telemetry_driver1 = fastest_driver1.get_car_data().add_distance()
    telemetry_driver2 = fastest_driver2.get_car_data().add_distance()
    delta_time, ref_tel, compare_tel = ff1.utils.delta_time(fastest_driver1, fastest_driver2)

    plt.rcParams['figure.figsize'] = [18, 15]
    fig, ax = plt.subplots(6, gridspec_kw={'height_ratios': [2, 3, 2, 1, 1, 2]})
    fig.suptitle(f"Fastest Lap Comparison, {driver1} vs. {driver2}\n{session_name}", size=20, fontweight='bold')
    ax[0].set_title(f"{driver1} Lap {format_laptime(fastest_driver1['LapTime'])} | {driver2} Lap {format_laptime(fastest_driver2['LapTime'])}", size=12, fontweight='bold')
    plt.subplots_adjust(top=0.91)

    # Time delta plot
    ax[0].plot(ref_tel['Distance'], delta_time, color='white', linewidth=1, alpha=0.5)
    ax[0].fill_between(ref_tel['Distance'], delta_time, 0, where=(delta_time > 0), color=color1, alpha=0.7, label=f"{driver1} faster")
    ax[0].fill_between(ref_tel['Distance'], delta_time, 0, where=(delta_time < 0), color=color2, alpha=0.7, label=f"{driver2} faster")
    ax[0].axhline(0, color='white', linestyle='--', linewidth=0.8)
    ax[0].set(ylabel=f"Gap to {driver1} (s)")
    ax[0].legend()

    # Speed plot
    ax[1].plot(telemetry_driver1['Distance'], telemetry_driver1['Speed'], label=driver1, color=color1)
    ax[1].plot(telemetry_driver2['Distance'], telemetry_driver2['Speed'], label=driver2, color=color2)
    ax[1].set(ylabel='Speed')
    ax[1].legend(loc="lower right")

    # Throttle plot
    ax[2].plot(telemetry_driver1['Distance'], telemetry_driver1['Throttle'], label=driver1, color=color1)
    ax[2].plot(telemetry_driver2['Distance'], telemetry_driver2['Throttle'], label=driver2, color=color2)
    ax[2].set(ylabel='Throttle')
    ax[2].legend(loc="lower right")

    # Brake plot
    ax[3].plot(telemetry_driver1['Distance'], telemetry_driver1['Brake'], label=driver1, color=color1)
    ax[3].plot(telemetry_driver2['Distance'], telemetry_driver2['Brake'], label=driver2, color=color2)
    ax[3].set(ylabel='Brake')
    ax[3].legend(loc="lower right")

    # Gear plot
    ax[4].plot(telemetry_driver1['Distance'], telemetry_driver1['nGear'], label=driver1, color=color1)
    ax[4].plot(telemetry_driver2['Distance'], telemetry_driver2['nGear'], label=driver2, color=color2)
    ax[4].set(ylabel='Gear')
    ax[4].legend(loc="lower right")

    # RPM plot
    ax[5].plot(telemetry_driver1['Distance'], telemetry_driver1['RPM'], label=driver1, color=color1)
    ax[5].plot(telemetry_driver2['Distance'], telemetry_driver2['RPM'], label=driver2, color=color2)
    ax[5].set(ylabel='RPM')
    ax[5].set(xlabel='Lap distance (meters)')
    ax[5].legend(loc="lower right")

    # Hide x-axis labels for all subplots except last
    for a in ax.flat:
        a.label_outer()

    # Label sectors
    sector1_dist = telemetry_driver1[telemetry_driver1['SessionTime'] >= fastest_driver1['Sector1SessionTime']].iloc[0]['Distance']
    sector2_dist = telemetry_driver1[telemetry_driver1['SessionTime'] >= fastest_driver1['Sector2SessionTime']].iloc[0]['Distance']

    for a in ax:
        a.axvline(x=sector1_dist, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
        a.axvline(x=sector2_dist, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    ax[0].text(sector1_dist, ax[0].get_ylim()[1], 'S2', color='white', fontsize=8)
    ax[0].text(sector2_dist, ax[0].get_ylim()[1], 'S3', color='white', fontsize=8)

    plt.show()

compound_colors = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFF200', 
    'HARD': '#EEEEEE',
    'INTERMEDIATE': '#39B54A',
    'WET': '#0067FF'
}

def plot_tire_strategy(session, regions=None):
    """

    session: FastF1 session object
    regions: Optional list of tuples (start_lap, end_lap, label, color, hatch) to highlight specific periods in the race
    """
    from math import ceil
    session_name = f"{session.event['EventName']} {session.event['EventDate'].year}"
    total_laps = session.laps['LapNumber'].max()

    # Compute the lap number range of each driver's stints, and include the tire compound used
    stints = session.laps.groupby(['Driver', 'Stint']).agg({'LapNumber':['min', 'max'], 'Compound':'first'}).reset_index()
    stints.columns = ['Driver', 'Stint', 'LapStart', 'LapEnd', 'Compound']
    finishing_positions = session.results['Abbreviation'].reset_index(drop=True)
    stints['Driver'] = pd.Categorical(stints['Driver'], categories=reversed(finishing_positions), ordered=True)
    stints = stints.sort_values(['Driver', 'Stint'])

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot the stacked bars for each tire compound and driver
    ax.barh(stints['Driver'], stints['LapEnd'] - stints['LapStart'] + 1, 
            left=stints['LapStart'], color=stints['Compound'].map(compound_colors))

    # Optionally highlight specific regions of the race
    if regions:
        for region in regions:
            ax.axvspan(region[0], region[1], color=region[3], hatch=region[4], alpha=0.5)

    # Plotting aesthetics
    legend_elements = [
        Patch(facecolor=compound_colors['SOFT'], label='Soft'),
        Patch(facecolor=compound_colors['MEDIUM'], label='Medium'),
        Patch(facecolor=compound_colors['HARD'], label='Hard'),
    ]
    legend_elements.extend([Patch(facecolor=region[3], alpha=0.5, hatch=region[4], label=region[2]) for region in regions])
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlabel('Lap Number', fontweight='bold')
    ax.set_title(f"Tire Strategy\n{session_name}", fontweight='bold')
    plt.xlim(0, ceil(total_laps / 10) * 10) # round up to lowest near 10 laps
    plt.show()