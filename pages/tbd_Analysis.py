import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

def analysis_dashboard():
    st.title("Material Management Analysis Dashboard")
    
    if 'output_df' not in st.session_state or st.session_state.output_df is None:
        st.warning("No analysis results available. Please run the analysis first.")
        return
    
    df = st.session_state.output_df.copy()
    
    # Convert relevant columns to appropriate types
    numeric_cols = [
        'Total Stock (k)', 'Max Stock Level', 'Min Stock Level',
        'Actual Reorder Level', 'Actual Reorder Quantity', 'Annual Demand', 
        'Demand Std Dev', 'Volume Value', 'Max', 'Min', 'Reorder Level', 
        'Reorder Quantity', 'Adjusted Max', 'Adjusted Min', 
        'Adjusted Reorder Level', 'Adjusted Reorder Qty'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            print(col)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Category filter
    if 'Category' in df.columns:
        categories = df['Category'].unique()
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=categories,
            default=categories
        )
        df = df[df['Category'].isin(selected_categories)]
    
    # Pareto Class filter
    if 'Pareto Class' in df.columns:
        pareto_classes = df['Pareto Class'].unique()
        selected_pareto = st.sidebar.multiselect(
            "Select Pareto Classes",
            options=pareto_classes,
            default=pareto_classes
        )
        df = df[df['Pareto Class'].isin(selected_pareto)]
    
    # Modify Category filter
    if 'Modified Category' in df.columns:
        mod_categories = df['Modified Category'].unique()
        selected_mod_categories = st.sidebar.multiselect(
            "Select Modified Categories",
            options=mod_categories,
            default=mod_categories
        )
        df = df[df['Modified Category'].isin(selected_mod_categories)]
    
    # Main content
    st.header("Filtered Data Overview")
    st.dataframe(df.head(100))  # Show first 100 rows to prevent overload
    
    # Statistics Section
    st.header("Key Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    if 'Annual Demand' in df.columns:
        col1.metric("Average Annual Consumption", f"{df['Annual Demand'].mean():.2f}")
        col2.metric("Median Annual Consumption", f"{df['Annual Demand'].median():.2f}")
        col3.metric("Total Annual Consumption", f"{df['Annual Demand'].sum():.2f}")
    
    if 'Volume Value' in df.columns:
        col1.metric("Average Volume", f"{df['Volume Value'].mean():.2f}")
        col2.metric("Median Volume", f"{df['Volume Value'].median():.2f}")
        col3.metric("Total Volume", f"{df['Volume Value'].sum():.2f}")
    
    # Visualization Section
    st.header("Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Relationships", "Categories", "Inventory Levels"])
    
    with tab1:
        st.subheader("Distribution of Key Metrics")
        
        dist_col = st.selectbox(
            "Select column for distribution analysis",
            options=numeric_cols,
            index=numeric_cols.index('Annual Demand') if 'Annual Demand' in numeric_cols else 0
        )
        
        fig = px.histogram(
            df, 
            x=dist_col,
            nbins=50,
            title=f"Distribution of {dist_col}",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative distribution
        fig = px.ecdf(
            df,
            x=dist_col,
            title=f"Cumulative Distribution of {dist_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Relationships Between Variables")
        
        x_axis = st.selectbox(
            "X-axis variable",
            options=numeric_cols,
            index=numeric_cols.index('Annual Demand') if 'Annual Demand' in numeric_cols else 0
        )
        
        y_axis = st.selectbox(
            "Y-axis variable",
            options=numeric_cols,
            index=numeric_cols.index('Volume Value') if 'Volume Value' in numeric_cols else 1
        )
        
        color_by = st.selectbox(
            "Color by",
            options=['Category', 'Pareto Class', 'Modified Category'],
            index=0
        ) if any(col in df.columns for col in ['Category', 'Pareto Class', 'Modified Category']) else None
        
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by if color_by in df.columns else None,
            hover_data=['Product Name', 'Number'],
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Between Numeric Variables",
                width=1000,
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Category Analysis")
        
        if 'Category' in df.columns:
            # Category distribution
            fig = px.pie(
                df,
                names='Category',
                title="Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category vs numeric metric
            metric = st.selectbox(
                "Select metric for category comparison",
                options=numeric_cols,
                index=numeric_cols.index('Annual Demand') if 'Annual Demand' in numeric_cols else 0
            )
            
            fig = px.box(
                df,
                x='Category',
                y=metric,
                title=f"{metric} by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Pareto Class' in df.columns:
            # Pareto class distribution
            fig = px.pie(
                df,
                names='Pareto Class',
                title="Distribution by Pareto Class"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Inventory Level Analysis")
        
        inventory_metrics = [
            col for col in [
                'Total Stock (k)', 'Max Stock Level', 'Min Stock Level',
                'Reorder Level', 'Reorder Quantity', 'Adjusted Max', 'Adjusted Min',
                'Adjusted Reorder Level', 'Adjusted Reorder Qty'
            ] if col in df.columns
        ]
        
        if inventory_metrics:
            selected_metrics = st.multiselect(
                "Select inventory metrics to compare",
                options=inventory_metrics,
                default=inventory_metrics[:2]
            )
            
            if selected_metrics:
                fig = go.Figure()
                for metric in selected_metrics:
                    fig.add_trace(go.Box(
                        y=df[metric],
                        name=metric,
                        boxpoints='all',
                        jitter=0.5,
                        whiskerwidth=0.2,
                        marker_size=2,
                        line_width=1)
                    )
                
                fig.update_layout(
                    title="Comparison of Inventory Metrics",
                    yaxis_title="Value",
                    boxmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series of inventory metrics (using annual as proxy)
                if 'Annual Demand' in df.columns:
                    fig = px.scatter(
                        df,
                        x='Annual Demand',
                        y=selected_metrics[0],
                        color=selected_metrics[1] if len(selected_metrics) > 1 else None,
                        title=f"{selected_metrics[0]} vs Annual Consumption",
                        hover_data=['Product Name', 'Number']
                    )
                    st.plotly_chart(fig, use_container_width=True)


# Run the dashboard
if __name__ == "__main__":
    analysis_dashboard()