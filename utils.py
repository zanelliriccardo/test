import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st

def load_data(stocks_file, consumption_file, prices_file, amos_file, sap_file):
    """Load and prepare all input data files."""
    # Stocks extraction
    stocks = pd.read_excel(stocks_file, sheet_name="Sheet6")
    
    # Consumption extraction
    consumption = pd.read_excel(consumption_file, sheet_name='consumption')
    consumption_frequency = pd.read_excel(
        consumption_file,
        sheet_name='frequency',
        dtype={'Row Labels': str}
    )

    prices = pd.read_excel(prices_file, sheet_name='price', dtype={'code': str})
    sap = pd.read_excel(sap_file, sheet_name='Sheet1', dtype={'Attrezzatura': str})
    amos = pd.read_excel(amos_file, sheet_name='Sheet1', dtype={'Number': str})
    
    return stocks, consumption, consumption_frequency, prices, sap, amos

def load_config(config_data):
    #func criticality mapping
    aux_config = config_data.parse('Function Criticality Weight')
    st.session_state['mapping_fc'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))
    print(st.session_state['mapping_fc'])

    #consumption mapping
    aux_config = config_data.parse('Consumption Weight')
    st.session_state['consumption_mapping'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[2]]))
    print(st.session_state['consumption_mapping'])

    #supply time mapping
    aux_config = config_data.parse('Median Supply Time Weight')
    st.session_state['supply_time_mapping'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[2]]))
    print(st.session_state['supply_time_mapping'])

    #criticality weight
    aux_config = config_data.parse('Criticality Computation Weight')
    st.session_state['criticality_weights'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))
    print(st.session_state['criticality_weights'])

    #criticality interval
    aux_config = config_data.parse('Criticality Class Interval')
    st.session_state['criticality_class_interval'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))
    print(st.session_state['criticality_class_interval'])

    #peso anni passati
    aux_config = config_data.parse('Years weight')
    st.session_state['past_years_weight'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[2]]))

    #external data like cost suppl time
    aux_config = config_data.parse('Fixed Costs')
    st.session_state['fixed_costs'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))

    #external data like cost suppl time
    aux_config = config_data.parse('Fixed Parameters')
    st.session_state['fixed_parameters'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))

    #service level
    aux_config = config_data.parse('Service level Matrix')
    st.session_state['service_level_matrix'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[1]]))

    
    #external data like cost suppl time
    aux_config = config_data.parse('Time Fix Supply Time')
    st.session_state['time_fix_supply_time'] = dict(zip(aux_config[aux_config.columns[0]], aux_config[aux_config.columns[2]]))

def clean_amos_data(amos):
    """Clean and standardize AMOS data."""
    if 'Component Number' in amos.columns:
        amos['Component Number'] = amos['Component Number'].astype(str)
        amos['Component Number'] = amos['Component Number'].str.replace("'", "")
        amos['Component Number'] = amos['Component Number'].str.replace(" ", "")
    else:
        amos['Func. No.'] = amos['Func. No.'].astype(str)
        amos['Func. No.'] = amos['Func. No.'].str.replace("'", "")
        amos['Func. No.'] = amos['Func. No.'].str.replace(" ", "")
        amos['Number'] = amos['Number'].astype(str)
        amos['Number'] = amos['Number'].str.replace("'", "")
        amos['Number'] = amos['Number'].str.replace(" ", "")
    return amos

def clean_sap_data(sap):
    """Clean and standardize SAP data."""
    sap['Attrezzatura'] = sap['Attrezzatura'].astype(str)
    sap['Attrezzatura'] = sap['Attrezzatura'].str.replace(r'\.0$', '', regex=True)
    return sap

def merge_datasets(sap, amos, consumption_frequency, stocks_copy):
    """Merge SAP, AMOS, consumption and stocks data."""
    sap_amos = (
        sap
        .merge(
            amos,
            left_on=['Attrezzatura'],
            right_on=['Component Number'] if 'Component Number' in amos.columns else ['Number'],
            how='outer'
        )
        .merge(
            (
                consumption_frequency[['Row Labels', 'Count of Quantity']]
                .merge(
                    stocks_copy[['Number', 'Median Supply time']],
                    left_on='Row Labels',
                    right_on='Number',
                    how='inner'
                )
            ),
            left_on='Materiale',
            right_on='Number',
            how='outer'
        )
        .fillna({'Count of Quantity': '0'})
        .fillna({'Func. Criticality': '0'})
    )
    return sap_amos

def calculate_criticality_weights(sap_amos, vessel_name):
    """
    Calculate functional criticality weights for items based on their criticality classification.
    
    This function maps each item's functional criticality to a numerical weight using a predefined mapping.
    The mapping is retrieved from the Streamlit session state. The function also validates that all
    criticality values have a corresponding mapping.
    
    Parameters:
    -----------
    sap_amos : pandas.DataFrame
        A DataFrame containing the SAP/AMOS data with at least the following column:
        - 'Func. Criticality': The functional criticality classification of each item
        
    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with an additional column:
        - 'fc_weight': The numerical weight assigned based on functional criticality
        
    Raises:
    -------
    ValueError
        If any items have functional criticality values that aren't present in the mapping dictionary
        
    Notes:
    ------
    - The mapping dictionary is expected to be stored in st.session_state['mapping_fc']
    - The function will raise an error if any unmapped criticality values are found
    """
    mapping_fc = st.session_state['mapping_fc']
    sap_amos['fc_weight'] = (
        sap_amos['Func. Criticality']
        .astype(str)
        .map(mapping_fc)
    )

    unmapped = sap_amos[~sap_amos['Func. Criticality'].isin(mapping_fc.keys())]
    if not unmapped.empty:
        raise ValueError(f"Unmapped criticality values: {unmapped['Func. Criticality'].unique()}")
    
    return sap_amos

def calculate_consumption_weights(sap_amos):
    """
    Calculate consumption weights for items based on their usage frequency.
    
    This function categorizes items into consumption frequency bins and assigns weights based on
    a predefined mapping. The mapping is retrieved from the Streamlit session state.
    
    Parameters:
    -----------
    sap_amos : pandas.DataFrame
        A DataFrame containing the SAP/AMOS data with at least the following column:
        - 'Count of Quantity': The consumption count for each item
        
    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with two additional columns:
        - 'consumption_category': The bin/category of consumption frequency
        - 'consumption_weight': The numerical weight assigned based on consumption frequency
        
    Notes:
    ------
    - The consumption mapping is expected to be stored in st.session_state['consumption_mapping']
    - Consumption categories are hardcoded as:
        ["0 times", "1-10 times", "11-20 times", "21-30 times", "up to 31 times"]
    - The 'Count of Quantity' column will be converted to numeric values
    """
    consumption_mapping = st.session_state['consumption_mapping']
    
    sap_amos['Count of Quantity'] = pd.to_numeric(sap_amos['Count of Quantity'], errors='raise')
    bins = [-1] + list(consumption_mapping.keys())[:-1] + [np.inf]
    labels = [0,1,11,21,31]

    sap_amos['consumption_category'] = pd.cut(
        sap_amos['Count of Quantity'],
        bins=bins,
        labels=labels,
    )

    sap_amos['consumption_weight'] = (
        sap_amos['consumption_category']
        .map(consumption_mapping)
        .astype(float)
        .fillna(0)
    )
    
    return sap_amos

def calculate_supply_time_weights(sap_amos):
    """
    Calculate supply time weights for items based on their median supply time.
    
    This function categorizes items into supply time duration bins and assigns weights based on
    a predefined mapping. The mapping is retrieved from the Streamlit session state.
    
    Parameters:
    -----------
    sap_amos : pandas.DataFrame
        A DataFrame containing the SAP/AMOS data with at least the following column:
        - 'Median Supply time': The median supply time in days for each item
        
    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with two additional columns:
        - 'supply_time_category': The bin/category of supply time duration
        - 'supply_time_weight': The numerical weight assigned based on supply time
        
    Notes:
    ------
    - The supply time mapping is expected to be stored in st.session_state['supply_time_mapping']
    - Supply time categories are hardcoded as:
        ["≤ 90 Days", "91 Days - 180 Days", "181 Days - 364 Days", "> 365 Days"]
    - The 'Median Supply time' column will be converted to numeric values
    """
    supply_time_mapping = st.session_state['supply_time_mapping']
    
    sap_amos['Median Supply time'] = pd.to_numeric(sap_amos['Median Supply time'], errors='raise')
    bins = [-1] + list(supply_time_mapping.keys())[:-1] + [np.inf]
    labels = [0,91,181,365]

    sap_amos['supply_time_category'] = pd.cut(
        sap_amos['Median Supply time'],
        bins=bins,
        labels=labels,
        right=True
    )

    sap_amos['supply_time_weight'] = (
        sap_amos['supply_time_category']
        .map(supply_time_mapping)
        .astype(float)
        .fillna(0)
    )
    return sap_amos

def calculate_criticality_score(sap_amos, vessel_name):
    """
    Calculate the final criticality score for items based on multiple weighted factors.
    
    This function combines the functional criticality, consumption, and supply time weights
    into a single criticality score using predefined weights. It then classifies items into
    criticality classes based on score thresholds.
    
    Parameters:
    -----------
    sap_amos : pandas.DataFrame
        A DataFrame containing the SAP/AMOS data with all the weight columns added by previous functions
    vessel_name : str
        The name of the vessel (used to determine the criticality ordering)
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the criticality analysis with the following columns:
        - Original identifying columns ('Number_x', 'Number_y')
        - Original factor columns ('Func. Criticality', 'Count of Quantity', 'Median Supply time')
        - Weight columns ('fc_weight', 'consumption_weight', 'supply_time_weight')
        - 'criticality': The calculated criticality score (0-1 scale)
        - 'class': The criticality class assigned based on score thresholds
        
    Notes:
    ------
    - The weights for each factor are expected to be stored in st.session_state['criticality_weights']
    - The criticality class intervals are expected to be stored in st.session_state['criticality_class_interval']
    - The function performs the following operations:
        1. Combines weights using weighted sum
        2. Sorts items by number and functional criticality
        3. Removes duplicates keeping the most critical item
        4. Classifies items into criticality classes based on thresholds
    - The criticality ordering is determined by the vessel name (though this is currently commented out)
    """
    weights = st.session_state['criticality_weights']
    
    sap_amos_criticality = (
        sap_amos[
            ['Number_x', 'Number_y', 'Func. Criticality', 'fc_weight', 
             'Count of Quantity', 'consumption_weight', 
             'Median Supply time', 'supply_time_weight']
        ]
        .assign(
            criticality=lambda df: (
                (df['fc_weight'] * weights["Function Criticality"]) + 
                (df['consumption_weight'] * weights["Consumption"]) + 
                (df['supply_time_weight'] * weights["Median Supply Time"])
            ).round(2)
        )
    )
    
    criticality_order = list(st.session_state['criticality_weights'].keys())
    
    sap_amos_criticality['Func. Criticality'] = pd.Categorical(
        sap_amos_criticality['Func. Criticality'],
        categories=criticality_order,
        ordered=True
    ).astype(str)

    sap_amos_criticality = sap_amos_criticality.sort_values(['Number_y', 'Func. Criticality'], ascending=[True, False])
    sap_amos_criticality = sap_amos_criticality.drop_duplicates('Number_y', keep='first')
    
    bins = [-1] + list(st.session_state['criticality_class_interval'].values())[::-1][1:] + [float('inf')]
    labels = list(st.session_state['criticality_class_interval'].keys())[::-1]
    
    sap_amos_criticality['class'] = (
        pd.cut(
            sap_amos_criticality['criticality'],
            bins=bins,
            labels=labels,
            right=False,
        ).astype('string')
    )
    
    return sap_amos_criticality

def merge_all_data(stocks, consumption, sap_amos_criticality, prices):
    """Merge all data sources into final dataframe."""
    print(stocks.columns)
    print(consumption.columns)
    print(sap_amos_criticality.columns)
    print(prices.columns)
    stocks['Number'] = stocks['Number'].astype('string')
    consumption['Row Labels'] = consumption['Row Labels'].astype('string')
    sap_amos_criticality['Number_y'] = sap_amos_criticality['Number_y'].astype('string')
    prices['code'] = prices['code'].astype('string')

    
    
    merged = (
        stocks
        .merge(
            consumption,
            left_on='Number',
            right_on='Row Labels',
            how='left'
        )
        .merge(
            sap_amos_criticality[['Number_y', 'fc_weight', 'consumption_weight', 'supply_time_weight', 'criticality', 'class']],
            left_on='Number',
            right_on='Number_y',
            how='left',
        )
        .merge(
            prices[['code', 'valuta', 'price']],
            left_on='Number',
            right_on='code',
            how='left'
        )
        .fillna({'class': 'Minor'})
    )

    
    return merged

def prepare_merged_data(merged):
    """Prepare and clean merged data with calculations."""
    merged['Number'] = merged['Number'].astype(str)
    merged[1] = merged[1].astype(float)
    merged[2] = merged[2].astype(float)
    merged[3] = merged[3].astype(float)
    merged[4] = merged[4].astype(float)
    merged[5] = merged[5].astype(float)

    merged['price'] = pd.to_numeric(merged['price'], errors='coerce')#.astype(float)
    merged['Stock Max.'] = merged['Stock Max.'] / 1000
    merged['Stock Min.'] = merged['Stock Min.'] / 1000
    merged['Reorder Level'] = merged['Reorder Level'] / 1000
    merged['Reorder Quantity'] = merged['Reorder Quantity'] / 1000

    merged = (
        merged
        .assign(
            outstanding = (
                merged['Outstanding Qty PO'] + 
                merged['Outstanding Qty TO']
            ) / 1000
        )
        .assign(
            stocks_plus_outstanding = (
                merged['total stocks/1000'] + 
                merged['Outstanding Qty PO']/1000 + 
                merged['Outstanding Qty TO']/1000
            )
        )
        .assign(
            adjusted_median_supply_time=merged['Median Supply time'].apply(
                lambda x: 330/365 if x > 330 else (30/365 if x < 30 else x/365)
            )
        )
        .fillna(
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
            }
        )
    )
    

    merged = (
        merged
        .assign(
            annual=(
                merged[1] * st.session_state['past_years_weight']['5 years ago'] +
                merged[2] * st.session_state['past_years_weight']['4 years ago'] +
                merged[3] * st.session_state['past_years_weight']['3 years ago'] +
                merged[4] * st.session_state['past_years_weight']['2 years ago'] +
                merged[5] * st.session_state['past_years_weight']['1 years ago']
            )
        )
        .assign(
            sdev=merged[[1, 2, 3, 4, 5]].std(axis=1, ddof=1)
        )
    )
    
    merged = merged.assign(
        volume_value = merged['price'] * merged['annual']
    )

    merged = (
        merged
        .assign(variability=merged['sdev'] / merged['annual'])
        .assign(
            variability_index=(
                merged['sdev'] / merged['annual'])
                .apply(
                    lambda x: "Z" if x <= 1 else ("Y" if 1 < x < 2 else "X")
                )
            )
        .fillna({
            'sdev':0,
            'variability_index':0,
        })
    )
    
    return merged

def calculate_service_level(merged):
    """Calculate service level based on class and variability."""
    threshold_dict = st.session_state['service_level_matrix']
    '''{
        "CriticalX": 0.90, "CriticalY": 0.85, "CriticalZ": 0.80,
        "MajorX": 0.90, "MajorY": 0.85, "MajorZ": 0.80,
        "NormalX": 0.80, "NormalY": 0.75, "NormalZ": 0.70,
        "MinorX": 0.75, "MinorY": 0.70, "MinorZ": 0.65,
        "TrivialX": 0.50, "TrivialY": 0.50, "TrivialZ": 0.50
    }'''

    merged = (
        merged
        .assign(
            class_variability=merged['class'] + merged['variability_index']
        )
    )
    merged = (
        merged
        .assign(
            service_level=merged['class_variability'].map(threshold_dict)
        )
        .assign(
            holding_cost = st.session_state['fixed_costs']['Holding Cost']
        )
        .assign(
            ordering_cost = st.session_state['fixed_costs']['Ordering Cost']
        )
    )
    
    return merged

def calculate_pareto_class(merged):
    """Calculate Pareto classification based on volume value."""
    merged = merged.sort_values(by='volume_value', ascending=False)
    merged = merged.assign(cumulative_volume_value=merged['volume_value'].cumsum() / merged['volume_value'].sum())
    
    merged = merged.assign(
        pareto_class=merged['cumulative_volume_value'].apply(
            lambda x: 'A' if x <= 0.80 else ('B' if x <= 0.95 else 'C')
        )
    )
    
    return merged

def categorize_items(merged):
    """
    Categorize inventory items into different management categories based on consumption patterns,
    stock levels, and Pareto classification.
    
    This function applies business rules to determine the appropriate inventory management strategy
    for each item by evaluating multiple conditions related to historical consumption, current stock
    levels, and item criticality (Pareto class).

    Parameters:
    -----------
    merged : pandas.DataFrame
        A DataFrame containing merged inventory data with the following required columns:
        - 'Name': Item name/description (used to identify deleted/obsolete items)
        - 1, 2, 3, 4, 5: Consumption quantities for the last 5 years (columns named 1-5)
        - 'total stocks/1000': Current stock levels (in thousands)
        - 'annual': Calculated annual demand
        - 'pareto_class': Pareto classification ('A', 'B', or 'C') based on volume value

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with an additional 'category' column containing the assigned
        inventory management category for each item. Possible categories are:
        - 'NO MRP': Items that should be excluded from MRP calculations
        - 'FROZEN': Items with stock but no recent consumption
        - 'EOQ': High-importance items suitable for Economic Order Quantity
        - 'TIME_FIX_B': Medium-importance items with fixed review period
        - 'TIME_FIX_C': Lower-importance items with fixed review period
        - 'UP_TO_MAX': Items to be replenished up to maximum stock level
        - 'NO CATEGORY': Default for items not matching any conditions

    Business Rules:
    --------------
    The categorization follows these hierarchical rules (evaluated in order):
    1. Items containing 'DELETED' or 'DO NOT USE' in name → 'NO MRP'
    2. Items with zero consumption for all 5 years AND zero stock → 'NO MRP'
    3. Items with zero consumption for all 5 years BUT positive stock → 'FROZEN'
    4. Items with annual demand ≥ 3 units:
       a. Pareto class A → 'EOQ' (most important items)
       b. Pareto class C → 'TIME_FIX_C' (least important high-demand items)
       c. Pareto class B → 'TIME_FIX_B' (medium importance items)
    5. Items with 0 < annual demand < 3 → 'UP_TO_MAX'
    6. All other items → 'NO CATEGORY'

    Notes:
    ------
    - The function uses numpy.select() for efficient conditional evaluation
    - Conditions are evaluated in the order listed above (first match wins)
    - The default 'NO CATEGORY' should be reviewed as it may indicate missing business rules
    - Consumption columns (1-5) should contain numerical values (zero for no consumption)
    - The 'pareto_class' should be pre-calculated based on volume value analysis

    Example:
    -------
    >>> inventory_data = pd.DataFrame({
    ...     'Name': ['Item1', 'DELETED_Item', 'Item3'],
    ...     1: [10, 0, 0],
    ...     2: [8, 0, 0],
    ...     3: [12, 0, 0],
    ...     4: [9, 0, 0],
    ...     5: [11, 0, 0],
    ...     'total stocks/1000': [5, 0, 2],
    ...     'annual': [4, 0, 0.5],
    ...     'pareto_class': ['A', 'C', 'B']
    ... })
    >>> categorized = categorize_items(inventory_data)
    >>> print(categorized[['Name', 'category']])
               Name   category
    0         Item1        EOQ
    1  DELETED_Item     NO MRP
    2         Item3  UP_TO_MAX
    """
    conditions = [
        merged['Name'].str.contains('DELETED|DO NOT USE', case=False, na=False),
        ((merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & 
         (merged[4] == 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] == 0)),
        ((merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & 
         (merged[4] == 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] > 0)),
        ((merged['annual'] >= 3) & (merged['pareto_class'] == "A")),
        ((merged['annual'] >= 3) & (merged['pareto_class'] == "C")),
        ((merged['annual'] >= 3) & (merged['pareto_class'] == "B")),
        ((merged['annual'] < 3) & (merged['annual'] > 0) & (merged['stocks_plus_outstanding'] > 0)),
        ((merged['annual'] < 3) & (merged['annual'] > 0) & (merged['stocks_plus_outstanding'] == 0)),
    ]

    choices = ["NO MRP", "NO MRP", "FROZEN", "EOQ", "TIME_FIX_C", "TIME_FIX_B", "UP_TO_MAX", "NO MRP"]

    merged['category'] = np.select(conditions, choices, default="NO CATEGORY")
    
    return merged

def calculate_inventory_parameters(merged):
    """
    Calculates core inventory management parameters (min, max, reorder levels) 
    based on item category and historical consumption patterns.
    
    This function implements different inventory policies according to ABC (Pareto)
    classification and each item's demand characteristics, using statistical
    inventory models tailored to each category.

    Parameters:
    -----------
    merged : pandas.DataFrame
        DataFrame containing inventory data with these mandatory columns:
        - 'category': Management category ('EOQ', 'TIME_FIX_B', 'TIME_FIX_C', etc.)
        - 'service_level': Target service level (0-1)
        - 'adjusted_median_supply_time': Lead time in years (capped between 30-330 days)
        - 'sdev': Standard deviation of annual demand
        - 'annual': Annual demand (weighted 5-year average)
        - 'ordering_cost': Fixed order cost
        - 'holding_cost': Annual holding cost rate
        - 'volume_value': Inventory value (price × annual demand)
        - 1,2,3,4,5: Historical consumption for last 5 years
        - 'stocks_plus_outstanding': Current stock + open orders


    Returns:
    --------
    pandas.DataFrame
        Input DataFrame enriched with these calculated columns:
        - 'min': Safety stock level
        - 'REORDER_LEVEL_intermediate': Intermediate reorder point (EOQ only)
        - 'REORDER_QTY_intermediate': Intermediate order quantity (EOQ only)
        - 'max': Maximum stock level
        - 'REORDER_LEVEL': Final reorder point
        - 'REORDER_QTY': Final order quantity

    Detailed Calculation Logic:
    --------------------------
    1. MIN (Safety Stock):
       - For EOQ items:
         Calculated as: 
         ceil(NORMSINV(service_level) × 
         √[(supply_time² × sdev²) + 
           (annual² × (supply_time × C44)²)]
         * NORMSINV: Inverse standard normal (Z-score)
         * Accounts for both demand variability and lead time uncertainty
       - Other categories: 0 (safety stock not calculated separately)

    2. INTERMEDIATE REORDER LEVEL (EOQ only):
       - Calculated as:
         ceil(annual × supply_time + min)
       * Represents lead time demand plus safety stock

    3. INTERMEDIATE ORDER QUANTITY (EOQ only):
       - Classical EOQ formula with adjustments:
         round(√[(2 × ordering_cost × annual) / 
                max(holding_cost × unit_value, C43)])
       * C43 ensures minimum holding cost consideration
       * Unit value = volume_value/annual

    4. MAX (Inventory Ceiling):
       - EOQ items:
         max(min + EOQ_quantity, 1.1 × reorder_level)
       - TIME_FIX_B items:
         ceil(annual × supply_time + 
              NORMSINV(service_level) × 
              √[(supply_time + B39)² × sdev² + 
                (supply_time × C44)² × annual²])
       - TIME_FIX_C items:
         Same as TIME_FIX_B but using C39 safety factor
       - UP_TO_MAX items:
         Mean of historical consumption (zeros excluded)
       - FROZEN items: Current stock position

    5. FINAL REORDER LEVEL:
       - EOQ: ceil(annual × supply_time + min)
       - TIME_FIX_B/C: 50% of max level
       - UP_TO_MAX: -1 (not applicable)
       - Others: 0

    6. FINAL ORDER QUANTITY:
       - EOQ: Recalculated EOQ
       - TIME_FIX_B/C: -1 (not applicable)
       - Others: 0

    Notes:
    ------
    - Negative values (-1) indicate "not applicable" for the category
    - All values are converted to integers (units)
    - TIME_FIX_B/C models use periodic review with safety factors
    - EOQ model assumes continuous review policy
    - Frozen items maintain current stock levels without replenishment

    Example:
    --------
    >>> config = {'C44': 0.1, 'C43': 5, 'B39': 30, 'C39': 60}
    >>> inventory = pd.DataFrame({
    ...     'category': ['EOQ', 'TIME_FIX_B', 'FROZEN'],
    ...     'service_level': [0.95, 0.90, 0.80],
    ...     'adjusted_median_supply_time': [0.2, 0.3, 0.1],
    ...     'sdev': [15, 10, 5],
    ...     'annual': [120, 80, 0],
    ...     'ordering_cost': [150, 150, 150],
    ...     'holding_cost': [0.25, 0.25, 0.25],
    ...     'volume_value': [6000, 4000, 0],
    ...     1: [20, 15, 0],  # Year 1 consumption
    ...     2: [18, 12, 0],  # Year 2
    ...     3: [25, 20, 0],  # Year 3
    ...     4: [22, 18, 0],  # Year 4
    ...     5: [15, 15, 0],  # Year 5
    ...     'stocks_plus_outstanding': [40, 30, 50]
    ... })
    >>> result = calculate_inventory_parameters(inventory, config)
    >>> print(result[['category', 'min', 'max', 'REORDER_LEVEL']])
    """
    # Calculate min stock
    conditions = [
        (merged['category'] == 'EOQ'),
        (merged['category'] == 'TIME_FIX_B'),
        (merged['category'] == 'TIME_FIX_C'),
        (merged['category'] == 'UP_TO_MAX'),
        (merged['category'] == 'NO MRP'),
        (merged['category'] == 'FROZEN')
    ]

    choices = [
        np.ceil(
            norm.ppf(merged['service_level']) 
        )
        * 
        np.sqrt(
            merged['adjusted_median_supply_time']**2 
            * 
            merged['sdev']**2 
            + 
            merged['annual']**2 
            * 
            (
                merged['adjusted_median_supply_time'] 
                * 
                st.session_state['fixed_parameters']["% dev standard rispetto al LT"]
            )**2
        ),
        0,
        0,
        0,
        0,
        0
    ]

    merged = merged.assign(
        min=np.select(conditions, choices, default=0)
    )

    # Calculate reorder level intermediate
    conditions = [
        (merged['category'] == 'EOQ'),
    ]

    choices = [
        np.ceil(
            merged['annual'] 
            * 
            merged['adjusted_median_supply_time'] 
            + 
            merged['min']
        ),
    ]

    merged = merged.assign(
        REORDER_LEVEL_intermediate=np.select(conditions, choices, default=0)
    )
    
    # Calculate reorder quantity intermediate
    conditions = [
        (merged['category'] == 'EOQ'),
    ]

    choices = [
        np.round(
            np.sqrt(
                2 
                * 
                merged['ordering_cost'] 
                * 
                merged['annual']
                / 
                np.where(
                    merged['holding_cost'] * merged['price']
                    < 
                    st.session_state['fixed_parameters']['min holding cost'], 
                    st.session_state['fixed_parameters']['min holding cost'], 
                    merged['price'] * merged['holding_cost']
                )
            ),
            0
        ).astype('Int64'),
    ]

    merged = merged.assign(
        REORDER_QTY_intermediate=np.select(conditions, choices, default=0)
    )
    
    # Calculate max stock
    conditions = [
        (merged['category'] == 'EOQ'),
        (merged['category'] == 'TIME_FIX_B'),
        (merged['category'] == 'TIME_FIX_C'),
        (merged['category'] == 'UP_TO_MAX'),
        (merged['category'] == 'NO MRP'),
        (merged['category'] == 'FROZEN')
    ]

    choices = [
        np.where(
            merged['min'] + merged['REORDER_QTY_intermediate'] < merged['REORDER_LEVEL_intermediate'],
            np.ceil(merged['REORDER_LEVEL_intermediate'] * 1.1),
            merged['min'] + merged['REORDER_QTY_intermediate']
        ),
        np.ceil(
            merged['adjusted_median_supply_time'] * merged['annual'] + 
            norm.ppf(merged['service_level']) * 
            np.sqrt(
                (merged['adjusted_median_supply_time'] + st.session_state['time_fix_supply_time']['B'] )**2 
                *
                (merged['sdev']**2)
                +   
                (st.session_state['fixed_parameters']['% dev standard rispetto al LT'] * merged['adjusted_median_supply_time'])**2 
                * 
                merged['annual']**2
            )
        ),
        np.ceil(
            merged['adjusted_median_supply_time'] * merged['annual'] + 
            norm.ppf(merged['service_level']) * 
            np.sqrt(
                (merged['adjusted_median_supply_time'] + st.session_state['time_fix_supply_time']['C'] )**2
                *
                (merged['sdev']**2)
                +  
                (st.session_state['fixed_parameters']['% dev standard rispetto al LT'] * merged['adjusted_median_supply_time'])**2 
                * 
                merged['annual']**2
            )
        ),
        merged[[1,2,3,4,5]].replace(0, np.nan).mean(axis=1),
        0,
        merged['stocks_plus_outstanding']
    ]

    merged = merged.assign(
        max=np.select(conditions, choices, default=0)
    )
    merged['max'] = merged['max'].astype(float)
    
    # Calculate reorder level
    conditions = [
        (merged['category'] == 'EOQ'),
        (merged['category'] == 'TIME_FIX_B'),
        (merged['category'] == 'TIME_FIX_C'),
        (merged['category'] == 'UP_TO_MAX'),
        (merged['category'] == 'NO MRP'),
        (merged['category'] == 'FROZEN')
    ]

    choices = [
        np.ceil(merged['annual'] * merged['adjusted_median_supply_time'] + merged['min']),
        0.5 * merged['max'],
        0.5 * merged['max'],
        -1,
        0,
        0
    ]

    merged = merged.assign(
        REORDER_LEVEL=np.select(conditions, choices, default=0)
    )
    
    # Calculate reorder quantity
    conditions = [
        (merged['category'] == 'EOQ'),
        (merged['category'] == 'TIME_FIX_B'),
        (merged['category'] == 'TIME_FIX_C'),
        (merged['category'] == 'UP_TO_MAX'),
        (merged['category'] == 'NO MRP'),
        (merged['category'] == 'FROZEN')
    ]

    choices = [
        np.round(
            np.sqrt(
                (2 * merged['ordering_cost'] * merged['annual']) 
                / 
                np.where(
                    (merged['holding_cost'] * merged['volume_value'] / merged['annual'] < st.session_state['fixed_parameters']['min holding cost']),
                    st.session_state['fixed_parameters']['min holding cost'],
                    merged['volume_value'] / merged['annual'] * merged['holding_cost']
                )
            ),
            0
        ),
        -1,
        -1,
        (merged['max'] - merged['stocks_plus_outstanding']),
        0,
        -1,
    ]

    merged = merged.assign(
        REORDER_QTY=np.select(conditions, choices, default=0)
    )
    
    return merged

def adjust_categories(merged):
    """Adjust categories based on additional business rules."""
    conditions = [
        merged['Name'].str.contains('DELETED|DO NOT USE', case=False, na=False),
        merged['category'] == 'NO MRP',
        (merged['Maker Name'].isin(['CATERPILLAR', 'STX CORPORATION'])) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] == 0),
        (merged['Maker Name'].isin(['CATERPILLAR', 'STX CORPORATION'])) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] != 0),
        (merged['Maker Name'].isin(['CATERPILLAR', 'STX CORPORATION'])) & (merged['max'] >= merged['stocks_plus_outstanding']),
        (merged['Maker Name'].isin(['CATERPILLAR', 'STX CORPORATION'])) & (merged['max'] < merged['stocks_plus_outstanding']),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] == 0) & (merged[5] == 0) & (merged['total stocks/1000'] == 0),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] == 0) & (merged[5] == 0),
        (merged[4] == 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] == 0),
        (merged[4] == 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] != 0),

        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] == 0) & (merged[5] > 0) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] == 0),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] == 0) & (merged[5] > 0) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] != 0),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] == 0) & (merged[5] > 0) & (merged['stocks_plus_outstanding'] != 0),

        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] > 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] == 0),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] > 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] == 0) & (merged['Stock Max.'] != 0),
        (merged[1] == 0) & (merged[2] == 0) & (merged[3] == 0) & (merged[4] > 0) & (merged[5] == 0) & (merged['stocks_plus_outstanding'] != 0),
        
        (merged['category'] == 'FROZEN') & (merged['class'].isin(['Minor', 'Trivial'])),
        (merged['category'] == 'FROZEN') & (merged['class'].isin(['Major', 'Normal', 'Critical'])),
    ]

    choices = [
        "NO MRP", 
        "NO MRP", 
        "NO MRP",
        "Caterpillar_0",
        "Caterpillar_1",
        "Caterpillar_1_1",
        "NO MRP",
        "FROZEN",
        "NO MRP",
        "UP_TO_MAX",

        "NO MRP",
        "LAST YEAR_1",
        "LAST YEAR",

        "NO MRP",
        "NO MRP",
        "Fourth_Year_1",

        "NO MRP",
        "UP_TO_MAX"
    ]

    merged = merged.assign(
        modify_category=np.select(conditions, choices, default=merged['category'])
    )
    
    return merged

def adjust_inventory_parameters(merged):
    """Adjust inventory parameters based on modified categories."""
    # Adjust min stock
    conditions = [
        (merged['modify_category'] == 'Caterpillar_0'),
        (merged['modify_category'] == 'NO MRP'),
        (merged['modify_category'] == 'UP_TO_MAX'),
        (merged['modify_category'] == 'Caterpillar_1'),
        (merged['modify_category'] == 'Caterpillar_1_1'),
        (merged['modify_category'] == 'Fourth_Year_1'),
        (merged['modify_category'] == 'LAST YEAR_1'),
        (merged['modify_category'] == 'LAST YEAR'),
    ]

    choices = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        np.select(
            [
                (merged['category'] == 'EOQ'),
                (merged['category'] == 'TIME_FIX_B'),
                (merged['category'] == 'TIME_FIX_C'),
                (merged['category'] == 'UP_TO_MAX'),
                (merged['category'] == 'NO MRP'),
            ],
            [
                np.ceil(
                    norm.ppf(merged['service_level'])
                ) 
                * 
                np.sqrt(
                    merged['adjusted_median_supply_time']**2 * (0.2*merged['annual'])**2 + 
                    merged[5]**2 * #prendere demand ultimo anno
                    (merged['adjusted_median_supply_time'] * st.session_state['fixed_parameters']['% dev standard rispetto al LT'])**2
                ),
                0,
                0,
                0,
                0,
            ],
            default=merged['min']
        )
    ]

    merged = merged.assign(
        adj_min=np.select(conditions, choices, default=merged['min'])
    )

    # Adjust reorder level intermediate
    conditions = [
        (merged['modify_category'] == 'LAST YEAR')
    ]

    choices = [
        np.select(
            [
                (merged['category'] == 'EOQ'),
            ],
            [
                np.ceil(merged['annual'] * merged['adjusted_median_supply_time'] + merged['adj_min']),
            ],
            default=merged['REORDER_LEVEL']
        )
    ]

    merged = merged.assign(
        adj_REORDER_LEVEL_intermediate=np.select(conditions, choices, default=merged['REORDER_LEVEL'])
    )

    # Adjust reorder quantity intermediate
    conditions = [
        (merged['modify_category'] == 'LAST YEAR') 
    ]

    choices = [
        np.select(
            [
                (merged['category'] == 'EOQ'),
            ],
            [
                np.ceil(
                    np.sqrt(
                        2 
                        * 
                        merged['ordering_cost'] 
                        * 
                        merged['annual']
                        / 
                        np.where(
                            merged['holding_cost'] * merged['price']
                            < 
                            st.session_state['fixed_parameters']['min holding cost'], 
                            st.session_state['fixed_parameters']['min holding cost'], 
                            merged['price'] * merged['holding_cost']
                        )
                    )
                )
            ],
            default=merged['REORDER_QTY']
        )
    ]

    merged = merged.assign(
        adj_REORDER_QTY_intermediate=np.select(conditions, choices, default=merged['REORDER_QTY'])
    )

    # Adjust max stock
    conditions = [
        (merged['modify_category'] == 'LAST YEAR_1'),
        (merged['modify_category'] == 'Caterpillar_0'),
        (merged['modify_category'] == 'NO MRP'),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN') & (merged['max'] < merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN') & (merged['max'] >= merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'Caterpillar_1'),
        (merged['modify_category'] == 'Caterpillar_1_1'),
        (merged['modify_category'] == 'Fourth_Year_1'),
        (merged['modify_category'] == 'LAST YEAR')
    ]

    choices = [
        merged['Stock Max.'],
        merged['Stock Max.'],
        0,
        merged['stocks_plus_outstanding'],
        merged['max'],
        merged['stocks_plus_outstanding'],
        merged['max'],
        merged['stocks_plus_outstanding'],
        np.select(
            [
                (merged['category'] == 'EOQ'),
                (merged['category'] == 'TIME_FIX_B'),
                (merged['category'] == 'TIME_FIX_C'),
                (merged['category'] == 'UP_TO_MAX') & (merged[5]>merged['stocks_plus_outstanding']),
                (merged['category'] == 'UP_TO_MAX') & (merged[5]<merged['stocks_plus_outstanding']),
                (merged['category'] == 'NO MRP'),
                (merged['category'] == 'On demand'),
                (merged['category'] == 'FROZEN')
            ],
            [
                np.where(
                    (merged['adj_min'] + merged['adj_REORDER_QTY_intermediate'] < merged['adj_REORDER_LEVEL_intermediate']),
                    np.ceil(merged['adj_REORDER_LEVEL_intermediate'] * 1.1),
                    merged['adj_REORDER_QTY_intermediate'] + merged['adj_min']
                ),
                np.ceil(
                    merged['adjusted_median_supply_time'] * merged['annual'] + 
                    norm.ppf(merged['service_level']) * 
                    np.sqrt(
                        (merged['adjusted_median_supply_time'] + st.session_state['time_fix_supply_time']['B'] )**2 
                        *
                        ((0.2*merged['annual'])**2)
                        +  
                        (st.session_state['fixed_parameters']['% dev standard rispetto al LT'] * merged['adjusted_median_supply_time'])**2 * 
                        merged[5]**2 # usare anche qua last year demand
                    )
                ),
                np.ceil(
                    merged['adjusted_median_supply_time'] * merged['annual'] + 
                    norm.ppf(merged['service_level']) * 
                    np.sqrt(
                        (merged['adjusted_median_supply_time'] + st.session_state['time_fix_supply_time']['C'] )**2
                        *
                        ((0.2*merged['annual'])**2)
                        +    
                        (st.session_state['fixed_parameters']['% dev standard rispetto al LT'] * merged['adjusted_median_supply_time'])**2 * 
                        merged[5]**2
                    )
                ),
                merged['stocks_plus_outstanding'],
                merged[5],
                0,
                0,
                0
            ],
            default=merged['max']
        )
    ]

    merged = merged.assign(
        adj_max=np.select(conditions, choices, default=merged['max'])
    )

    # Calculate final max
    conditions = [
        (merged['modify_category'] == 'LAST YEAR') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN')  & (merged['adj_max'] > merged['stocks_plus_outstanding']), #riscrivo meglio
        (merged['modify_category'] == 'Fourth_Year_1') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
    ]

    choices = [
        merged['stocks_plus_outstanding'],
        merged['stocks_plus_outstanding'],
        merged['stocks_plus_outstanding'],
    ]

    merged = merged.assign(
        final_max=np.select(conditions, choices, default=merged['adj_max'])
    )

    # Calculate final min
    conditions = [
        (merged['modify_category'] == 'LAST YEAR') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN')  & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'Fourth_Year_1') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
    ]

    choices = [
        0,
        0,
        0,
    ]

    merged = merged.assign(
        final_min=np.select(conditions, choices, default=merged['adj_min'])
    )
    
    # Adjust reorder level
    conditions = [
        (merged['modify_category'] == 'LAST YEAR_1'),
        (merged['modify_category'] == 'Caterpillar_0'),
        (merged['modify_category'] == 'NO MRP'),
        (merged['modify_category'] == 'UP_TO_MAX'),
        (merged['modify_category'] == 'Caterpillar_1'),
        (merged['modify_category'] == 'Caterpillar_1_1'),
        (merged['modify_category'] == 'Fourth_Year_1'),
        (merged['modify_category'] == 'LAST YEAR'),
    ]

    choices = [
        merged['REORDER_LEVEL'],
        -1,
        0,
        -1,
        -1,
        merged['REORDER_LEVEL'],
        -1,
        np.select(
            [
                (merged['category'] == 'EOQ'),
                (merged['category'] == 'TIME_FIX_B'),
                (merged['category'] == 'TIME_FIX_C'),
                (merged['category'] == 'UP_TO_MAX'),
                (merged['category'] == 'NO MRP'),
                (merged['category'] == 'On demand')
            ],
            [
                np.ceil(merged['annual'] * merged['adjusted_median_supply_time'] + merged['adj_min']),
                0.5*merged['final_max'],
                0.5*merged['final_max'],
                -1,
                -1,
                -1
            ],
            default=merged['REORDER_LEVEL']
        )
    ]

    merged = merged.assign(
        adj_REORDER_LEVEL=np.select(conditions, choices, default=merged['REORDER_LEVEL'])
    )

    # Calculate final reorder level
    conditions = [
        (merged['modify_category'] == 'LAST YEAR') ,#& (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN')  & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'Fourth_Year_1') ,#& (merged['adj_max'] > merged['stocks_plus_outstanding']),
    ]

    choices = [
        -1,
        -1,
        -1,
    ]

    merged = merged.assign(
        final_REORDER_LEVEL=np.select(conditions, choices, default=merged['adj_REORDER_LEVEL'])
    )

    # Adjust reorder quantity
    conditions = [
        (merged['modify_category'] == 'Caterpillar_0'),
        (merged['modify_category'] == 'NO MRP'),
        (merged['modify_category'] == 'UP_TO_MAX'),
        (merged['modify_category'] == 'Caterpillar_1'),
        (merged['modify_category'] == 'Caterpillar_1_1'),
        (merged['modify_category'] == 'Fourth_Year_1'),
        (merged['modify_category'] == 'LAST YEAR') | (merged['modify_category'] == 'LAST YEAR_1')
    ]

    choices = [
        -1,
        0,
        -1,
        -1,
        merged['REORDER_QTY'],
        -1,
        np.select(
            [
                (merged['category'] == 'EOQ'),
                (merged['category'] == 'TIME_FIX_B'),
                (merged['category'] == 'TIME_FIX_C'),
                (merged['category'] == 'UP_TO_MAX'),
                (merged['category'] == 'NO MRP'),
                (merged['category'] == 'On demand')
            ],
            [
                -1,#merged['final_max'] - merged['stocks_plus_outstanding'],
                -1,
                -1,
                -1,
                -1,
                -1
            ],
            default=merged['REORDER_QTY']
        )
    ]

    merged = merged.assign(
        adj_REORDER_QTY=np.select(conditions, choices, default=merged['REORDER_QTY'])
    )

    # Calculate final reorder quantity
    conditions = [
        (merged['modify_category'] == 'LAST YEAR') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'UP_TO_MAX') & (merged['category'] != 'FROZEN')  & (merged['adj_max'] > merged['stocks_plus_outstanding']),
        (merged['modify_category'] == 'Fourth_Year_1') & (merged['adj_max'] > merged['stocks_plus_outstanding']),
    ]

    choices = [
        -1,
        -1,
        -1,
    ]

    merged = merged.assign(
        final_REORDER_QTY=np.select(conditions, choices, default=merged['adj_REORDER_QTY'])
    )

    # Round final values
    merged['final_max'] = round(merged['final_max'], 0)
    merged['final_min'] = round(merged['final_min'], 0)
    merged['final_REORDER_QTY'] = round(merged['final_REORDER_QTY'], 0)
    merged['final_REORDER_LEVEL'] = round(merged['final_REORDER_LEVEL'], 0)
    
    return merged

def calculate_alignment_metrics(merged):
    """Calculate under-reorder quantities and alignment costs."""
    merged = (
        merged
        .assign(
            under_reorder=np.select(
                [
                    (merged['modify_category'].isin(['LAST YEAR_1'])),
                    (merged['category'] == 'UP_TO_MAX') & (merged['stocks_plus_outstanding'] == 0),
                    merged['final_REORDER_LEVEL'] > 0,
                    True
                ],
                [
                    0,
                    0,
                    merged['final_REORDER_LEVEL'] - (merged['stocks_plus_outstanding']),
                    merged['final_max'] - (merged['stocks_plus_outstanding'])
                ],
                default=np.nan
            )
        )
    )

    merged = (
        merged.assign(
            alignment_cost=np.select(
                [
                    (merged['under_reorder'] > 0) & (merged['category'] == 'EOQ'),
                    (merged['under_reorder'] > 0) & (merged['category'] != 'EOQ'),
                    True
                ],
                [
                    merged['final_REORDER_QTY'] * merged['price'],
                    (merged['final_max'] - (merged['stocks_plus_outstanding'])) * merged['price'],
                    0
                ]
            )
        )
    )
    
    return merged

def map_showed_categories(merged):
    """Map modify_category to showed_category with business rules."""
    category_mapping = {
        'EOQ': 'EOQ',
        'TIME_FIXED_B': 'TIME_FIXED',
        'TIME_FIXED_C': 'TIME_FIXED',
        'NO MRP': 'NO MRP',
        'FROZEN': 'FROZEN',
        'UP_TO_MAX': 'UP_TO_MAX',
        'Caterpillar_0': 'CATERPILLAR_UP_TO_MAX',
        'Caterpillar_1': 'CATERPILLAR_UP_TO_MAX',
        'Caterpillar_1_1': None, #filled after with Caterpillar + previous category
        'LAST YEAR_1': None,
        'LAST YEAR': 'LAST YEAR',
        'Fourth_Year_1': None
    }

    merged['showed_category'] = np.where(
        merged['modify_category'] == 'Caterpillar_1_1',
        'CATERPILLAR_' + merged['category'],
        merged['modify_category'].map(category_mapping).fillna(merged['modify_category'])
    )
    merged['showed_category'] = np.where(
        merged['modify_category'] == 'Caterpillar_0',
        'CATERPILLAR_' + merged['category'],
        merged['showed_category'].map(category_mapping).fillna(merged['showed_category'])
    )
    merged['showed_category'] = np.where(
        merged['modify_category'] == 'LAST YEAR_1',
        'LAST YEAR_1_' + merged['category'],
        merged['showed_category'].map(category_mapping).fillna(merged['showed_category'])
    )
    merged['showed_category'] = np.where(
        merged['modify_category'] == 'Fourth_Year_1',
        'Fourth_Year_1' + merged['category'],
        merged['showed_category'].map(category_mapping).fillna(merged['showed_category'])
    )
    
    return merged

def prepare_final_output(merged):
    """Prepare the final output dataframe with selected columns and renamed headers."""
    output = (
        merged
        [[
            'Number',
            'Name',
            'Maker Name',
            'total stocks/1000',
            'outstanding',
            'stocks_plus_outstanding',
            'Stock Max.',
            'Stock Min.',
            'Reorder Level',
            'Reorder Quantity',
            'adjusted_median_supply_time',
            1,
            2,
            3,
            4,
            5,
            'annual',
            'sdev',
            'variability',
            'variability_index',
            'class',
            'service_level',
            'holding_cost',
            'ordering_cost',
            'volume_value',
            'pareto_class',
            'category',
            'max',
            'min',
            'REORDER_LEVEL',
            'REORDER_QTY',
            'modify_category',
            'showed_category',
            'final_max',
            'final_min',
            'final_REORDER_LEVEL',
            'final_REORDER_QTY',
            'price',
            'under_reorder',
            'alignment_cost',
        ]]
        .sort_index()
    )

    # Replace -1 with '-' for display purposes
    output[[
        'REORDER_LEVEL', 'REORDER_QTY', 'final_REORDER_LEVEL', 'final_REORDER_QTY'
    ]] = output[[
        'REORDER_LEVEL', 'REORDER_QTY', 'final_REORDER_LEVEL', 'final_REORDER_QTY'
        ]].replace(-1, '-')
    
    # Rename columns for final output
    output = (
        output
        .rename(
            columns={
                'Number': 'Number',
                'Name': 'Product Name',
                'Maker Name': 'Manufacturer',
                'total stocks/1000': 'Total Stock (k)',
                'outstanding': 'Outstanding Orders',
                'stocks_plus_outstanding': 'Total Inventory',
                'Stock Max.': 'Max Stock Level',
                'Stock Min.': 'Min Stock Level',
                'Reorder Level': 'Actual Reorder Level',
                'Reorder Quantity': 'Actual Reorder Quantity',
                'adjusted_median_supply_time': 'Median Supply Time (adj)',
                1: '5 years ago demand',
                2: '4 years ago demand',
                3: '3 years ago demand',
                4: '2 years ago demand',
                5: '1 years ago demand',
                'annual': 'Annual Demand',
                'sdev': 'Demand Std Dev',
                'variability': 'Demand Variability',
                'variability_index': 'Variability Index',
                'class': 'Product Class',
                'service_level': 'Service Level',
                'holding_cost': 'Holding Cost',
                'ordering_cost': 'Ordering Cost',
                'volume_value': 'Volume Value',
                'pareto_class': 'Pareto Class',
                'category': 'Category',
                'max': 'Max',
                'min': 'Min',
                'REORDER_LEVEL': 'Reorder Level',
                'REORDER_QTY': 'Reorder Quantity',
                'modify_category': 'Modified Category',
                'showed_category': 'Showed Category',
                'final_max': 'Adjusted Max',
                'final_min': 'Adjusted Min',
                'final_REORDER_LEVEL': 'Adjusted Reorder Level',
                'final_REORDER_QTY': 'Adjusted Reorder Qty',
                'under_reorder': 'Under Reorder Quantity',
                'alignment_cost': 'Alignment Cost',
            }
        )
    )
    
    return output

