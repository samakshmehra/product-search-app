import streamlit as st
import json
import pandas as pd
from dotenv import load_dotenv
import os
from match import (
    join_product_context,
    product_type_detection,
    search_products_with_embeddings,
    refine_search_results_with_llm
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Product Search & Refinement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-header {
        font-size: 1.8rem;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #10b981, #059669);
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: none;
    }
    .manufacturer-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .seller-card {
        background-color: white;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff7f0e;
    }
    .product-card {
        background-color: #f8f9fa;
        padding: 0.6rem;
        border-radius: 0.2rem;
        margin: 0.3rem 0;
        border-left: 2px solid #2ca02c;
    }
    .similarity-score {
        color: #d62728;
        font-weight: bold;
    }
    .exact-match {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .related-product {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .alternative-brand {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 0px 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Product Search & Refinement</h1>', unsafe_allow_html=True)
    
    # Compact input fields in two rows
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        product_name = st.text_input("Product Name", placeholder="e.g., Digital Current Monitoring Relay", key="main_product_name")
    with col2:
        description = st.text_input("Description", placeholder="Description (optional)", key="main_description")
    with col3:
        specification = st.text_input("Specification", placeholder="Specification (optional)", key="main_specification")
    with col4:
        manufacture_name = st.text_input("Manufacturer", placeholder="Manufacturer (optional)", key="main_manufacturer")
    
    # Advanced settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        col5, col6, col7, col8 = st.columns([1, 1, 1, 1])
        with col5:
            manufacturer_weight = st.slider(
                "Manufacturer Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.1, 
                help="Weight for manufacturer matching (0.0 = ignore manufacturer, 1.0 = prioritize manufacturer)",
                key="main_manufacturer_weight"
            )
        with col6:
            threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.05, 
                help="Minimum similarity score (0.0 = all results, 1.0 = exact matches only)",
                key="main_threshold"
            )
        with col7:
            limit = st.number_input(
                "Max Results", 
                min_value=10, 
                max_value=500, 
                value=100, 
                step=10, 
                help="Maximum number of results to process",
                key="main_limit"
            )
        with col8:
            ai_filtering = st.checkbox("🤖 AI Filtering", value=True, help="Enable AI refinement to get categorized results", key="main_ai_filtering")
    
    # Search button
    search_button = st.button("🔍 Search & Refine", type="primary", use_container_width=True, key="main_search_button")

    # Main content area
    if search_button and product_name:
        # Permanent step status display
        st.markdown("## 🔄 Processing Steps")
        
        # Create a permanent status container
        status_container = st.container()
        with status_container:
            step1_status = st.empty()
            step2_status = st.empty()
            step3_status = st.empty()
        
        def update_step_status(current_step_num, step_name, status="⏳"):
            """Update the permanent step status display"""
            steps = [
                (1, "Detecting Product Type"),
                (2, "Searching Database"),
                (3, "LLM Categorization" if ai_filtering else "Grouping Results")
            ]
            
            status_elements = [step1_status, step2_status, step3_status]
            
            for i, (step_num, step_text) in enumerate(steps):
                if step_num < current_step_num:
                    if step_num != 1:
                        status_elements[i].markdown(f"✅ **Step {step_num}: {step_text}** - Completed")
                elif step_num == current_step_num:
                    status_elements[i].markdown(f"{status} **Step {step_num}: {step_text}** - {step_name}")
                else:
                    status_elements[i].empty()
        
        try:
            if ai_filtering:
                # Step-by-step process with AI refinement
                update_step_status(1, "Sending product details to LLM for type detection...")
                context_parts = join_product_context(product_name, description, specification)
                detected_product_type = product_type_detection(context_parts)
                
                # Update step 1 to show detected product type
                step1_status.markdown(f"✅ **Step 1: Detecting Product Type** - Product Type: {detected_product_type}")
                
                update_step_status(2, "Searching database for matching products...")
                products = search_products_with_embeddings(
                    product_type=detected_product_type,
                    product_name=product_name,
                    manufacture_name=manufacture_name if manufacture_name else None,
                    manufacturer_weight=manufacturer_weight,
                    threshold=threshold,
                    limit=limit
                )
                
                step2_status.markdown(f"✅ **Step 2: Searching Database** - Found {len(products)} products")
                
                update_step_status(3, "Sending results to LLM for categorization...")
                
                if products:
                    results_json = json.dumps(products)
                    results = refine_search_results_with_llm(
                        search_results_json=results_json,
                        product_name=product_name,
                        detected_product_type=detected_product_type,
                        description=description if description else None,
                        specification=specification if specification else None,
                        manufacture_name=manufacture_name if manufacture_name else None
                    )
                    
                    step3_status.markdown("✅ **Step 3: LLM Categorization** - Completed")
                else:
                    results = None
                    step3_status.markdown("⚠️ **Step 3: LLM Categorization** - Skipped (no products found)")
            else:
                # Perform search without AI refinement
                update_step_status(1, "Sending product details to LLM for type detection...")
                context_parts = join_product_context(product_name, description, specification)
                detected_product_type = product_type_detection(context_parts)
                
                step1_status.markdown(f"✅ **Step 1: Detecting Product Type** - Product Type: {detected_product_type}")
                
                update_step_status(2, "Searching database for matching products...")
                products = search_products_with_embeddings(
                    product_type=detected_product_type,
                    product_name=product_name,
                    manufacture_name=manufacture_name if manufacture_name else None,
                    manufacturer_weight=manufacturer_weight,
                    threshold=threshold,
                    limit=limit
                )
                
                step2_status.markdown(f"✅ **Step 2: Searching Database** - Found {len(products)} products")
                
                update_step_status(3, f"Grouping {len(products)} products by manufacturer...")
                results = products  # Simple grouping without AI
                step3_status.markdown("✅ **Step 3: Grouping Results** - Completed")
            
            st.success("✅ Search completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            st.info("💡 Make sure your environment variables (GEMINI_API_KEY, POSTGRESQL_URL) are set correctly.")
            return
        
        # Display results
        display_results(results, product_name, ai_filtering)
    
    elif search_button and not product_name:
        st.error("❌ Please enter a product name to search")
    
    else:
        # Welcome message
        st.markdown("""
        ## 🚀 Welcome to Product Search & Refinement
        
        This advanced tool helps you find and categorize similar products across different manufacturers.
        
        ### How to use:
        1. **Enter Product Name** (required) - The main product you're looking for
        2. **Add Optional Details** - Description, specifications, and preferred manufacturer  
        3. **Adjust Advanced Settings** - Fine-tune search parameters
        4. **Enable AI Filtering** - Get intelligent categorization of results
        5. **Click Search** - View organized results
        
        ### Features:
        - 🎯 **Smart Categorization** - AI-powered result organization
        - 🔍 **Advanced Search** - Vector similarity + manufacturer weighting
        - 📊 **Rich Display** - Detailed product information in organized tables
        - ⚙️ **Customizable** - Adjustable similarity thresholds and result limits
        """)

def display_results(results, product_name, ai_filtering=True):
    """Display results with categorized output for AI filtering, or simple list for non-AI"""
    st.markdown(f"## 📊 Search Results for: {product_name}")
    
    if not results:
        st.info("No results found.")
        return
    
    if ai_filtering and hasattr(results, 'products'):
        # Display categorized results from LLM
        display_categorized_results(results, product_name)
    else:
        # Display simple product list
        display_simple_results(results, product_name)

def display_categorized_results(results, product_name: str):
    """Display LLM-categorized results in tabs"""
    
    # Categorize products
    same_manu_same_prod = [p for p in results.products if p.category == "same_manufacturer_same_product"]
    same_manu_diff_prod = [p for p in results.products if p.category == "same_manufacturer_different_products"]
    diff_manu_same_prod = [p for p in results.products if p.category == "different_manufacturer_same_product"]
    
    # Summary
    st.markdown(f"**Total Products Found: {len(results.products)}**")
    
    # Create tabs for each category
    tab1, tab2, tab3 = st.tabs([
        f"🎯 Exact Matches ({len(same_manu_same_prod)})",
        f"🔄 Related Products ({len(same_manu_diff_prod)})",
        f"🏢 Alternative Brands ({len(diff_manu_same_prod)})"
    ])
    
    with tab1:
        if same_manu_same_prod:
            display_product_table(same_manu_same_prod, "exact-match")
        else:
            st.info("No exact matches found.")
    
    with tab2:
        if same_manu_diff_prod:
            display_product_table(same_manu_diff_prod, "related-product")
        else:
            st.info("No related products from the same manufacturer found.")
    
    with tab3:
        if diff_manu_same_prod:
            display_product_table(diff_manu_same_prod, "alternative-brand")
        else:
            st.info("No alternative brands for the same product type found.")

def display_simple_results(products, product_name: str):
    """Display simple product list without categorization"""
    st.markdown(f"**Total Products Found: {len(products)}**")
    
    if products:
        display_product_table(products, "product-card")
    else:
        st.info("No products found.")

def display_product_table(products, css_class=""):
    """Display products in a single flat table with manufacturer and seller columns"""
    
    # Create table data for all products
    table_rows = []
    for product in products:
        # Handle both Pydantic objects and dictionaries
        if hasattr(product, 'product_name'):
            # Pydantic object
            embedding_score = getattr(product, 'embedding_similarity_score', 0)
            llm_score = getattr(product, 'llm_similarity_score', None)
            table_rows.append({
                "Manufacturer": product.manufacture_name or 'Unknown',
                "Seller": product.seller_name or 'Unknown',
                "Product Name": product.product_name,
                "Product Type": product.product_type or '',
                "Embedding Score": f"{embedding_score:.3f}",
                "LLM Score": f"{llm_score:.3f}" if llm_score is not None else "N/A",
                "Code": product.product_code or '',
                "Description": (product.product_description or '')[:80] + "..." if product.product_description and len(product.product_description) > 80 else (product.product_description or ''),
                "Specification": (product.product_specification or '')[:80] + "..." if product.product_specification and len(product.product_specification) > 80 else (product.product_specification or '')
            })
        else:
            # Dictionary - check for both possible field names
            embedding_score = product.get('embedding_similarity_score', product.get('similarity_score', 0))
            llm_score = product.get('llm_similarity_score')
            table_rows.append({
                "Manufacturer": product.get('manufacture_name', 'Unknown'),
                "Seller": product.get('seller_name', 'Unknown'),
                "Product Name": product.get('product_name', ''),
                "Product Type": product.get('product_type', '') or '',
                "Embedding Score": f"{embedding_score:.3f}",
                "LLM Score": f"{llm_score:.3f}" if llm_score is not None else "N/A",
                "Code": product.get('product_code', '') or '',
                "Description": (product.get('product_description', '') or '')[:80] + "..." if product.get('product_description') and len(product.get('product_description', '')) > 80 else (product.get('product_description', '') or ''),
                "Specification": (product.get('product_specification', '') or '')[:80] + "..." if product.get('product_specification') and len(product.get('product_specification', '')) > 80 else (product.get('product_specification', '') or '')
            })
    
    if table_rows:
        df = pd.DataFrame(table_rows)
        
        # Display the single flat table
        st.dataframe(
            df,
            use_container_width=True,
            height=min(600, len(table_rows) * 35 + 50),
            column_config={
                "Manufacturer": st.column_config.TextColumn(
                    "Manufacturer",
                    width="medium"
                ),
                "Seller": st.column_config.TextColumn(
                    "Seller",
                    width="medium"
                ),
                "Product Name": st.column_config.TextColumn(
                    "Product Name",
                    width="large"
                ),
                "Product Type": st.column_config.TextColumn(
                    "Product Type",
                    width="small"
                ),
                "Embedding Score": st.column_config.NumberColumn(
                    "Embedding Score",
                    format="%.3f",
                    width="small"
                ),
                "LLM Score": st.column_config.TextColumn(
                    "LLM Score",
                    width="small"
                ),
                "Code": st.column_config.TextColumn(
                    "Code",
                    width="small"
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large"
                ),
                "Specification": st.column_config.TextColumn(
                    "Specification",
                    width="large"
                )
            }
        )
    else:
        st.info("No products found.")
    
    # Add spacing between sellers
    st.markdown("---")

if __name__ == "__main__":
    # Check environment variables
    if not os.getenv('GEMINI_API_KEY'):
        st.error("❌ GEMINI_API_KEY not found in environment variables")
        st.stop()
    
    if not os.getenv('POSTGRESQL_URL'):
        st.error("❌ POSTGRESQL_URL not found in environment variables")
        st.stop()
    
    main()
