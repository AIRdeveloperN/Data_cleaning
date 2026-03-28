import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

st.set_page_config(page_title="Data Cleaning App", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if "transformation_log" not in st.session_state:
    st.session_state.transformation_log = []
if "history" not in st.session_state:
    st.session_state.history = []
if "working_df" not in st.session_state:
    st.session_state.working_df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None


def log_transformation(operation, details=""):
    st.session_state.transformation_log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "details": details
    })


@st.cache_data
def load_data(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "csv":
        return pd.read_csv(uploaded_file)
    elif file_type == "xlsx":
        return pd.read_excel(uploaded_file)
    elif file_type == "json":
        return pd.read_json(uploaded_file)
    else:
        return None


def page_upload():
    st.title("Upload and Overview")

    if "u_key" not in st.session_state:
        st.session_state.u_key = 0

    col1, col2 = st.columns([6, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Select File", 
            type=["csv", "xlsx", "json"],
            key=f"loader_{st.session_state.u_key}"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear()
            st.cache_data.clear()
            st.session_state.u_key = np.random.randint(1, 1000)
            st.rerun()

    if uploaded_file is not None:
        if st.session_state.get("file_name") != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.original_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.file_name = uploaded_file.name
                st.session_state.transformation_log = [] 
                st.success(f"Loaded file: {uploaded_file.name}")


    if st.session_state.working_df is not None:
        df = st.session_state.working_df

        st.subheader("Dataset Overview")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", df.shape[0])
        m2.metric("Columns", df.shape[1])
        m3.metric("Missing Values", int(df.isnull().sum().sum()))
        m4.metric("Duplicates", int(df.duplicated().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Count": df.isnull().sum().values,
            "Unique Values": [df[col].nunique() for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Numeric Summary")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        with st.expander("Transformation Log"):
            if st.session_state.transformation_log:
                st.dataframe(pd.DataFrame(st.session_state.transformation_log), use_container_width=True)
            else:
                st.write("No transformations yet")


def page_cleaning():
    st.title(" Cleaning Studio")

    if "working_df" not in st.session_state or st.session_state.working_df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.working_df

    st.subheader("Current Working Data")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

    tabs = st.tabs([
        "Missing Values",
        "Duplicates",
        "Data Types",
        "Categorical",
        "Numerical",
        "Scaling"
    ])

    with tabs[0]:
        st.subheader("Handle Missing Values")
        
        df_missing = st.session_state.working_df

        null_counts = df_missing.isnull().sum()
        if null_counts.sum() == 0:
            st.success("No missing values found!")
        else:
            summary_df = pd.DataFrame({
                "Column": null_counts.index,
                "Missing Values": null_counts.values
            })
            st.dataframe(summary_df[summary_df["Missing Values"] > 0], use_container_width=True)

        st.divider()

        st.write("### Drop Columns by Threshold")
        threshold = st.slider("Threshold (%)", 0, 100, 50)
        if st.button("Apply Threshold Drop"):
            limit = len(df_missing) * (threshold / 100)
            cols_to_drop = df_missing.columns[df_missing.isnull().sum() > limit].tolist()
            
            if cols_to_drop:
                st.session_state.working_df = df_missing.drop(columns=cols_to_drop)
                log_transformation("Drop Threshold", f"Dropped: {cols_to_drop}")
                st.success(f"Dropped: {cols_to_drop}")
                st.rerun()
            else:
                st.info("No columns exceed this threshold.")

        st.divider()

        st.write("### Specific Column Imputation")
        selected_col = st.selectbox("Select Column", df_missing.columns)
        
        current_nulls = int(df_missing[selected_col].isnull().sum())
        st.write(f"Missing values in **{selected_col}**: {current_nulls}")

        method = st.selectbox("Select Method", 
                             ["Drop Rows", "Fill with Mean", "Fill with Median", 
                              "Fill with Mode", "Fill with Custom Value", 
                              "Forward Fill", "Backward Fill"])

        custom_val = ""
        if method == "Fill with Custom Value":
            custom_val = st.text_input("Enter Value")

        if st.button("Clean Column"):
            if current_nulls == 0:
                st.warning("This column is already clean.")
            else:
                new_df = df_missing.copy()
                
                try:
                    if method == "Drop Rows":
                        new_df = new_df.dropna(subset=[selected_col])
                    elif method == "Fill with Mean":
                        new_df[selected_col] = new_df[selected_col].fillna(pd.to_numeric(new_df[selected_col], errors='coerce').mean())
                    elif method == "Fill with Median":
                        new_df[selected_col] = new_df[selected_col].fillna(pd.to_numeric(new_df[selected_col], errors='coerce').median())
                    elif method == "Fill with Mode":
                        new_df[selected_col] = new_df[selected_col].fillna(new_df[selected_col].mode()[0])
                    elif method == "Fill with Custom Value":
                        new_df[selected_col] = new_df[selected_col].fillna(custom_val)
                    elif method == "Forward Fill":
                        new_df[selected_col] = new_df[selected_col].ffill()
                    elif method == "Backward Fill":
                        new_df[selected_col] = new_df[selected_col].bfill()

                    st.session_state.working_df = new_df
                    log_transformation("Missing Values", f"{method} on {selected_col}")
                    st.success(f"Applied {method} successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}. Check if the column is numeric for Mean/Median.")

    with tabs[1]:
        st.subheader("Duplicate Cleanup Studio")
        
        df_dup = st.session_state.working_df
        st.write("### 1. How should I find duplicates?")
        dup_mode = st.radio("Search mode:", ["Whole Row (Exact Match)", "By Specific Columns"])
        
        selected_cols = None
        if dup_mode == "By Specific Columns":
            selected_cols = st.multiselect("Select columns to check:", df_dup.columns.tolist())
            if not selected_cols:
                st.info("Please select at least one column to start searching.")
                return
        
        duplicate_rows = df_dup[df_dup.duplicated(subset=selected_cols, keep=False)]
        
        st.metric("Duplicates Found", len(duplicate_rows))

        if len(duplicate_rows) > 0:
            with st.expander("View all duplicate groups"):
                st.dataframe(duplicate_rows.sort_values(by=selected_cols if selected_cols else df_dup.columns[0]), use_container_width=True)

            st.write("### 2. Which one should I keep?")
            keep_choice = st.selectbox("Keep option:", ["first", "last"])
            
            if st.button("Remove Duplicates Now"):
                clean_df = df_dup.drop_duplicates(subset=selected_cols, keep=keep_choice)
                removed_count = len(df_dup) - len(clean_df)
                st.session_state.working_df = clean_df
                log_transformation("Duplicates", f"Removed {removed_count} duplicates (kept {keep_choice})")
                st.success(f"Done! {removed_count} duplicate rows were removed.")
                st.rerun()
        else:
            st.success("Clean! No duplicates found with these settings.")

    with tabs[2]:
        st.subheader("Data Types & Parsing")
        df_dtype = st.session_state.working_df
        col_to_convert = st.selectbox("Select column to convert:", df_dtype.columns)
        target_type = st.selectbox("Convert to:", ["Numeric", "Categorical", "Datetime"])

        date_format = None
        if target_type == "Datetime":
            st.info("💡 Hint: Y=Year, m=Month, d=Day")
            date_format = st.selectbox("Select Date Format (or Auto):", 
                                     ["Auto Detect", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"])

        if st.button("Apply Conversion"):
            temp_df = df_dtype.copy()
            try:
                if target_type == "Numeric":
                    temp_df[col_to_convert] = pd.to_numeric(temp_df[col_to_convert], errors="coerce")
                elif target_type == "Categorical":
                    temp_df[col_to_convert] = temp_df[col_to_convert].astype("category")
                elif target_type == "Datetime":
                    if date_format == "Auto Detect":
                        temp_df[col_to_convert] = pd.to_datetime(temp_df[col_to_convert], errors="coerce")
                    else:
                        temp_df[col_to_convert] = pd.to_datetime(temp_df[col_to_convert], format=date_format, errors="coerce")

                st.session_state.working_df = temp_df
                log_transformation("Dtype Change", f"{col_to_convert} to {target_type}")
                st.success(f"Column '{col_to_convert}' is now {target_type}!")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {e}")

        st.write("#### Current Column Types:")
        st.write(df_dtype.dtypes.to_frame(name="Data Type").T)

    with tabs[3]:
        st.subheader("Categorical Tools & Mapping")
        df_cat = st.session_state.working_df
        cat_columns = df_cat.select_dtypes(include=['object']).columns.tolist()
        
        if not cat_columns:
            st.info("No categorical columns found in the dataset.")
        else:
            selected_col = st.selectbox("Select Categorical Column", cat_columns)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Trim Whitespace"):
                    df_cat[selected_col] = df_cat[selected_col].astype(str).str.strip()
                    st.session_state.working_df = df_cat
                    log_transformation("Categorical", f"Trimmed whitespace in {selected_col}")
                    st.success("Whitespaces removed!")
                    st.rerun()
            
            with col2:
                if st.button("Convert to Lowercase"):
                    df_cat[selected_col] = df_cat[selected_col].str.lower()
                    st.session_state.working_df = df_cat
                    log_transformation("Categorical", f"Lowercased {selected_col}")
                    st.success("Converted to lowercase!")
                    st.rerun()
            
            with col3:
                if st.button("Convert to Uppercase"):
                    df_cat[selected_col]= df_cat[selected_col].str.upper()
                    st.session_state.working_df = df_cat
                    log_transformation("Categorical",f"Uppercased{selected_col}")
                    st.success("Coverted to uppercase!")
                    st.rerun()

            st.divider()
            st.write("### 2. Value Mapping")
            unique_vals = df_cat[selected_col].unique()
            map_df = pd.DataFrame({"Current": unique_vals, "New": unique_vals})
            edited_map = st.data_editor(map_df, use_container_width=True, key="editor_cat")
            
            if st.button("Apply Mapping"):
                mapping_dict = dict(zip(edited_map["Current"], edited_map["New"]))
                df_cat[selected_col] = df_cat[selected_col].map(mapping_dict).fillna(df_cat[selected_col])
                st.session_state.working_df = df_cat
                log_transformation("Mapping", f"Mapped {selected_col}")
                st.rerun()

            st.divider()
            st.write("### 3. Rare Category Grouping")
            threshold = st.slider("Frequency Threshold (%)", 0, 50, 5)
            if st.button("Group Rare Values"):
                counts = df_cat[selected_col].value_counts(normalize=True) * 100
                rare_values = counts[counts < threshold].index
                df_cat[selected_col] = df_cat[selected_col].replace(rare_values, "Other")
                st.session_state.working_df = df_cat
                log_transformation("Rare Grouping", f"Grouped values < {threshold}% in {selected_col}")
                st.rerun()

            st.divider()
            st.write("### 4. One-Hot Encoding")
            if st.button("Apply One-Hot Encoding"):
                encoded_df = pd.get_dummies(df_cat, columns=[selected_col], prefix=selected_col)
                st.session_state.working_df = encoded_df
                log_transformation("Encoding", f"One-hot encoded {selected_col}")
                st.success(f"{selected_col} converted to multiple binary columns!")
                st.rerun()

    with tabs[4]:
        st.subheader("Numeric Cleaning: Outliers")
        df_num = st.session_state.working_df
        num_cols = df_num.select_dtypes(include=['number']).columns.tolist()

        if not num_cols:
            st.info("No numeric columns found.")
        else:
            selected_col = st.selectbox("Select Numeric Column", num_cols)
            
            Q1 = df_num[selected_col].quantile(0.25)
            Q3 = df_num[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_num[(df_num[selected_col] < lower_bound) | (df_num[selected_col] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Bound", f"{lower_bound:.2f}")
            col2.metric("Upper Bound", f"{upper_bound:.2f}")
            col3.metric("Outliers Found", len(outliers))

            action = st.radio("Action:", ["Do Nothing", "Cap/Winsorize", "Remove Outlier Rows"])

            if st.button("Apply Outlier Action"):
                if action == "Do Nothing":
                    st.warning("No action selected.")
                else:
                    new_df = df_num.copy()
                    old_len = len(new_df)

                    if action == "Cap/Winsorize":
                        new_df[selected_col] = new_df[selected_col].clip(lower=lower_bound, upper=upper_bound)
                        impact = f"Capped values to [{lower_bound:.2f}, {upper_bound:.2f}]"
                    elif action == "Remove Outlier Rows":
                        new_df = new_df[(new_df[selected_col] >= lower_bound) & (new_df[selected_col] <= upper_bound)]
                        impact = f"Removed {old_len - len(new_df)} outlier rows"

                    st.session_state.working_df = new_df
                    log_transformation("Outliers", f"{action} on {selected_col}")
                    st.success(f"Success! {impact}")
                    st.rerun()

        st.divider()
        st.write("### Data Preview")
        st.dataframe(st.session_state.working_df.head(10), use_container_width=True)

    with tabs[5]:
        st.subheader("Normalization & Scaling")
        df_scale = st.session_state.working_df
        num_cols = df_scale.select_dtypes(include=['number']).columns.tolist()

        if not num_cols:
            st.info("No numeric columns found.")
        else:
            selected_cols = st.multiselect("Select columns to scale:", num_cols)
            method = st.selectbox("Choose method:", ["Min-Max Scaling", "Z-Score Standardization"])

            if selected_cols:
                st.write("#### Stats Before Scaling:")
                st.dataframe(df_scale[selected_cols].describe().T[['min', 'max', 'mean', 'std']], use_container_width=True)

                if st.button("Apply Scaling"):
                    new_df = df_scale.copy()
                    for col in selected_cols:
                        if method == "Min-Max Scaling":
                            c_min, c_max = new_df[col].min(), new_df[col].max()
                            if c_max != c_min:
                                new_df[col] = (new_df[col] - c_min) / (c_max - c_min)
                        elif method == "Z-Score Standardization":
                            c_mean, c_std = new_df[col].mean(), new_df[col].std()
                            if c_std != 0:
                                new_df[col] = (new_df[col] - c_mean) / c_std

                    st.session_state.working_df = new_df
                    log_transformation("Scaling", f"{method} on {selected_cols}")
                    
                    st.success(f"Successfully applied {method}!")
                    st.write("#### Stats After Scaling:")
                    st.dataframe(new_df[selected_cols].describe().T[['min', 'max', 'mean', 'std']], use_container_width=True)
                    st.rerun()
            else:
                st.warning("Please select at least one column.")

def page_visualization():
    st.title("📊 Page C: Visualization Builder")

    if st.session_state.working_df is None:
        st.warning("Please upload a data file first.")
        return

    df = st.session_state.working_df
    
    st.sidebar.subheader("Quick Filters")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    filtered_df = df.copy()
    
    if num_cols:
        f_col = st.sidebar.selectbox("Filter by Number:", num_cols)
        min_v = float(df[f_col].min())
        max_v = float(df[f_col].max())
        
        if min_v < max_v:
            val_range = st.sidebar.slider(f"Select range for {f_col}", min_v, max_v, (min_v, max_v))
            filtered_df = filtered_df[(filtered_df[f_col] >= val_range[0]) & (filtered_df[f_col] <= val_range[1])]
        else:
            st.sidebar.info(f"{f_col} ustunida barcha qiymatlar bir xil ({min_v}).")

    if cat_cols:
        f_cat = st.sidebar.selectbox("Filter by Category:", ["All"] + cat_cols)
        if f_cat != "All":
            unique_vals = df[f_cat].dropna().unique()
            selected_val = st.sidebar.multiselect(f"Select {f_cat}", unique_vals)
            if selected_val:
                filtered_df = filtered_df[filtered_df[f_cat].isin(selected_val)]

    st.subheader("Chart Settings")
    chart_type = st.selectbox("Choose chart type", 
        ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Correlation Heatmap"])

    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "Histogram":
        col = st.selectbox("Select numeric column", num_cols)
        ax.hist(filtered_df[col].dropna(), bins=20, color='green', edgecolor='white')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    elif chart_type == "Box Plot":
        col = st.selectbox("Select numeric column", num_cols)
        plot_data = filtered_df[col].dropna()
        if not plot_data.empty:
            ax.boxplot(plot_data, patch_artist=True, 
                       boxprops=dict(facecolor='orange', color='black'))
            ax.set_title(f"Boxplot of {col}")
            ax.set_ylabel(col)
        else:
            st.warning("Ma'lumot mavjud emas.")

    elif chart_type == "Scatter Plot":
        x_axis = st.selectbox("X Axis (Numeric)", num_cols)
        y_axis = st.selectbox("Y Axis (Numeric)", num_cols, key="y_scat")
        ax.scatter(filtered_df[x_axis], filtered_df[y_axis], alpha=0.6, color='royalblue')
        ax.set_title(f"{x_axis} vs {y_axis}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.grid(True, linestyle='--', alpha=0.5)

    elif chart_type == "Line Chart":
        x_axis = st.selectbox("X Axis", df.columns)
        y_axis = st.selectbox("Y Axis (Numeric)", num_cols, key="y_line")
        ax.plot(filtered_df[x_axis], filtered_df[y_axis], marker='o', markersize=4)
        ax.set_title(f"{y_axis} over {x_axis}")
        plt.setp(ax.get_xticklabels(), rotation=45)

    elif chart_type == "Bar Chart":
        x_axis = st.selectbox("Category Column", cat_cols)
        y_axis = st.selectbox("Value Column", num_cols)
        agg_type = st.selectbox("Calculate as", ["mean", "sum", "count"])
        top_n = st.slider("Show Top N categories", 5, 20, 10)
        
        chart_data = filtered_df.groupby(x_axis)[y_axis].agg(agg_type).sort_values(ascending=False).head(top_n)
        chart_data.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_ylabel(agg_type.capitalize())
        plt.setp(ax.get_xticklabels(), rotation=45)

    elif chart_type == "Correlation Heatmap":
        if len(num_cols) > 1:
            corr_matrix = filtered_df[num_cols].corr()
            if not corr_matrix.isnull().all().all():
                im = ax.imshow(corr_matrix, cmap='coolwarm')
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(len(num_cols)))
                ax.set_yticks(range(len(num_cols)))
                ax.set_xticklabels(num_cols, rotation=90)
                ax.set_yticklabels(num_cols)
                for i in range(len(num_cols)):
                    for j in range(len(num_cols)):
                        val = corr_matrix.iloc[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
            else:
                st.warning("Korrelyatsiyani hisoblash uchun yetarli ma'lumot yo'q.")
        else:
            st.warning("Heatmap uchun kamida 2 ta raqamli ustun kerak.")

    st.pyplot(fig)

import streamlit as st
import pandas as pd
import io

def page_export():
    st.title("📤 Step 3: Export & Report")

    if "working_df" not in st.session_state or st.session_state.working_df is None:
        st.warning("⚠️ Please upload and process a data file first.")
        return

    df = st.session_state.working_df

    st.subheader("📋 Transformation Report")
    
    if st.session_state.get('transformation_log'):
        log_df = pd.DataFrame(st.session_state.transformation_log)
        st.dataframe(log_df, use_container_width=True)
        
        log_json = log_df.to_json(orient='records', indent=4)

        col_log1, col_log2 = st.columns(2)
        
        with col_log1:
            st.download_button(
                label="📥 Download Log (CSV)",
                data=log_df.to_csv(index=False).encode('utf-8'),
                file_name="transformation_log.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_log2:
            st.download_button(
                label="📥 Download Log (JSON)",
                data=log_json,
                file_name="transformation_log.json",
                mime="application/json",
                use_container_width=True
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Undo Last Step"):
                if st.session_state.get('history'):
                    st.session_state.working_df = st.session_state.history.pop()
                    st.session_state.transformation_log.pop()
                    st.success("Last change reverted.")
                    st.rerun()
                else:
                    st.info("No more steps to undo.")
        
        with col2:
            if st.button("♻️ Reset All Changes"):
                if st.session_state.get('original_df') is not None:
                    st.session_state.working_df = st.session_state.original_df.copy()
                    st.session_state.transformation_log = []
                    st.session_state.history = []
                    st.success("Dataset reset to original state.")
                    st.rerun()
    else:
        st.info("No transformations recorded in this session.")

    st.divider()

    st.subheader("🧮 Quick Aggregation")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()

    if numeric_cols:
        try:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                group_col = st.selectbox("Group By", all_cols, key="agg_group")
            with col_b:
                value_col = st.selectbox("Value Column", numeric_cols, key="agg_val")
            with col_c:
                agg_func = st.selectbox("Function", ["mean", "sum", "count", "min", "max"], key="agg_func")

            res_df = df.groupby(group_col)[value_col].agg(agg_func).reset_index()
            st.dataframe(res_df, use_container_width=True)

            agg_csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Aggregation (CSV)", agg_csv, "summary.csv", "text/csv")
            
        except Exception as e:
            st.error(f"❌ Aggregation failed: {e}")
    else:
        st.warning("No numeric columns found for aggregation.")

    st.divider()

    st.subheader("💾 Download Full Cleaned Dataset")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        csv_final = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download as CSV",
            data=csv_final,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_out2:
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='CleanedData')
            
            st.download_button(
                label="Excel (XLSX)",
                data=buffer.getvalue(),
                file_name="cleaned_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception:
            st.info("Excel export requires 'openpyxl' library.")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Upload ",
            "Cleaning Studio",
            "Visualization Builder",
            "Export"
        ])

    if page == "Upload ":
        page_upload()
    elif page == "Cleaning Studio":
        page_cleaning()
    elif page == "Visualization Builder":
        page_visualization()
    else:
        page_export()

    if st.session_state.working_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.write(f"File: {st.session_state.file_name}")
        st.sidebar.write(
            f"Shape: {st.session_state.working_df.shape[0]} rows × {st.session_state.working_df.shape[1]} columns"
        )
        st.sidebar.write(f"Transformations: {len(st.session_state.transformation_log)}")

if __name__ == "__main__":
    main()