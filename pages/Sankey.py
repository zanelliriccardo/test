# pages/sankey_analysis.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show_sankey_analysis():
    st.set_page_config(page_title="Inventory Flow Analysis", layout="wide")
    st.title("🔀 Inventory Dimension Relationships")
    
    if 'output_df' not in st.session_state or st.session_state.output_df is None:
        st.warning("Please run the main analysis first to load data")
        return
    
    df = st.session_state.output_df.copy()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Sankey Diagram Controls")
        
        selected_columns = st.multiselect(
            "Select dimensions to visualize",
            options=['Variability Index', 'Product Class', 'Pareto Class', 'Category', 'Modified Category'],
            default=['Variability Index', 'Product Class', 'Pareto Class', 'Category'],
            max_selections=5
        )
        
        if len(selected_columns) < 2:
            st.error("Please select at least 2 dimensions")
            return
        
        st.subheader("Filters")
        if 'Product Class' in df.columns:
            class_filter = st.multiselect(
                "Filter by Product Class",
                options=df['Product Class'].unique(),
                default=df['Product Class'].unique()
            )
            df = df[df['Product Class'].isin(class_filter)]
        
        if 'Pareto Class' in df.columns:
            pareto_filter = st.multiselect(
                "Filter by Pareto Class",
                options=df['Pareto Class'].unique(),
                default=df['Pareto Class'].unique()
            )
            df = df[df['Pareto Class'].isin(pareto_filter)]
    
    # Generate Sankey data
    def prepare_sankey_data(df, columns):
        # Create all combinations of consecutive columns
        links = pd.DataFrame()
        
        for i in range(len(columns)-1):
            source_col = columns[i]
            target_col = columns[i+1]
            
            # Count combinations
            combos = df.groupby([source_col, target_col]).size().reset_index(name='value')
            combos['source'] = combos[source_col]
            combos['target'] = combos[target_col]
            combos['step'] = i
            
            links = pd.concat([links, combos])
        
        # Create node list
        unique_sources = links['source'].unique()
        unique_targets = links['target'].unique()
        all_nodes = pd.concat([pd.Series(unique_sources), pd.Series(unique_targets)]).unique()
        
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Map to indices
        links['source_idx'] = links['source'].map(node_dict)
        links['target_idx'] = links['target'].map(node_dict)
        
        return {
            'nodes': list(all_nodes),
            'links': links
        }
    
    sankey_data = prepare_sankey_data(df, selected_columns)
    
    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['nodes'],
            color="blue"
        ),
        link=dict(
            source=sankey_data['links']['source_idx'],
            target=sankey_data['links']['target_idx'],
            value=sankey_data['links']['value'],
            hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title_text="Inventory Flow Between Dimensions",
        font_size=12,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data summary
    st.subheader("Relationship Summary")
    
    for i in range(len(selected_columns)-1):
        source_col = selected_columns[i]
        target_col = selected_columns[i+1]
        
        st.write(f"#### {source_col} → {target_col}")
        
        crosstab = pd.crosstab(
            df[source_col],
            df[target_col],
            margins=True,
            margins_name="Total"
        )
        
        st.dataframe(
            crosstab.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

if __name__ == "__main__":
    show_sankey_analysis()