#!/usr/bin/env python3
"""
Streamlit App for Product Search and Refinement
"""

import streamlit as st
import json
import pandas as pd
import os
from matchingcorrectedcode import refine_search_results_with_llm, search_products_with_embeddings, group_products_by_manufacturer, join_product_context, product_type_detection, CategorizedResults

# Function to get environment variables (works both locally and on Streamlit Cloud)
def get_env_var(key):
    """Get environment variable from st.secrets (Streamlit Cloud) or os.environ (local)"""
    try:
        # Try st.secrets first (Streamlit Cloud)
        return st.secrets[key]
    except:
        # Fallback to os.environ (local development)
        return os.getenv(key)

# Page config
st.set_page_config(
    page_title="Product Search & Refinement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Product Search & Refinement</h1>', unsafe_allow_html=True)
    
    # Compact input fields in two rows
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    with col1:
        product_name = st.text_input("Product Name", placeholder="e.g., Digital Current Monitoring Relay", label_visibility="collapsed")
    with col2:
        description = st.text_input("Description", placeholder="Description (optional)", label_visibility="collapsed")
    with col3:
        specification = st.text_input("Specification", placeholder="Specification (optional)", label_visibility="collapsed")
    with col4:
        manufacture_name = st.text_input("Manufacturer", placeholder="Manufacturer (optional)", label_visibility="collapsed")
    
    # Advanced settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        col5, col6, col7, col8 = st.columns([1, 1, 1, 1])
        with col5:
            manufacturer_weight = st.slider("Manufacturer Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="Weight for manufacturer matching (0.0 = ignore manufacturer, 1.0 = prioritize manufacturer)")
        with col6:
            threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05, help="Minimum similarity score (0.0 = all results, 1.0 = exact matches only)")
        with col7:
            limit = st.number_input("Max Results", min_value=10, max_value=500, value=100, step=10, help="Maximum number of results to process")
        with col8:
            ai_filtering = st.checkbox("🤖 AI Filtering", value=True, help="Enable AI refinement to get categorized results")
    
    # Search button
    col8 = st.columns([1])[0]
    with col8:
        search_button = st.button("🔍 Search & Refine", type="primary")
    
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
                    # Skip step 1 if it already has custom product type display
                    if step_num != 1:
                        status_elements[i].markdown(f"✅ **Step {step_num}: {step_text}** - Completed")
                elif step_num == current_step_num:
                    status_elements[i].markdown(f"{status} **Step {step_num}: {step_text}** - {step_name}")
                else:
                    # Don't show future steps
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
                
                grouped_results = group_products_by_manufacturer(products, top_products_per_seller=10)
                
                update_step_status(3, "Sending grouped results to LLM for categorization...")
                results_json = json.dumps(grouped_results, indent=2, default=str)
                
                results = refine_search_results_with_llm(
                    context_parts=context_parts,
                    detected_product_type=detected_product_type,
                    manufacture_name=manufacture_name if manufacture_name else None,
                    results_json=results_json
                )
                
                # Mark all steps as complete
                update_step_status(4, "All steps completed!", "✅")
            else:
                # Perform search without AI refinement
                update_step_status(1, "Sending product details to LLM for type detection...")
                context_parts = join_product_context(product_name, description, specification)
                detected_product_type = product_type_detection(context_parts)
                
                # Update step 1 to show detected product type
                step1_status.markdown(f"✅ **Step 1: Detecting Product Type** - Product Type: {detected_product_type}")
                
                update_step_status(2, "Searching database for matching products...")
                products = search_products_with_embeddings(
                    product_type=detected_product_type,
                    product_name=product_name,
                    manufacture_name=manufacture_name,
                    manufacturer_weight=manufacturer_weight,
                    threshold=threshold,
                    limit=limit
                )
                
                update_step_status(3, f"Found {len(products)} products, grouping by manufacturer...")
                grouped_results = group_products_by_manufacturer(products, top_products_per_seller=10)
                
                # Convert to old format for backward compatibility
                results = grouped_results
                
                # Mark all steps as complete
                update_step_status(4, "All steps completed!", "✅")
            
            st.success("✅ Search completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            st.info("💡 Make sure your environment variables (GEMINI_API_KEY, POSTGRESQL_URL) are set correctly.")
            return
        
        # Display results
        display_results(results, product_name, ai_filtering)
    
    elif not search_button:
        # Simple welcome message
        st.markdown("""
        ## Welcome to Product Search
        
        Enter a product name and click Search to find matching products.
        """)

def display_results(results, product_name, ai_filtering=True):
    """Display results with categorized output for AI filtering, or grouped results for non-AI"""
    st.markdown(f"## Search Results for: {product_name}")
    
    if not results:
        st.info("No results found.")
        return
    
    if ai_filtering and isinstance(results, CategorizedResults):
        # Display categorized results
        display_categorized_results(results, product_name)
    else:
        # Display grouped results (backward compatibility)
        display_grouped_results(results, product_name)

def display_categorized_results(results: CategorizedResults, product_name: str):
    """Display results in three categories using tabs"""
    
    # Summary statistics
    total_products = (
        sum(len(seller.products) for manufacturer in results.same_manufacturer_same_product for seller in manufacturer.sellers) +
        sum(len(seller.products) for manufacturer in results.same_manufacturer_different_products for seller in manufacturer.sellers) +
        sum(len(seller.products) for manufacturer in results.different_manufacturer_same_product for seller in manufacturer.sellers)
    )
    
    st.markdown(f"**Total Products Found: {total_products}**")
    
    # Create tabs for each category
    tab1, tab2, tab3 = st.tabs([
        "🎯 Exact Matches: Same Manufacturer, Same Product",
        "🔄 Related Products: Same Manufacturer, Different Product",
        "🏢 Alternative Brands: Different Manufacturer, Same Product"
    ])
    
    with tab1:
        if results.same_manufacturer_same_product:
            display_manufacturer_groups(results.same_manufacturer_same_product, "exact-match")
        else:
            st.info("No exact matches found.")
            
    with tab2:
        if results.same_manufacturer_different_products:
            display_manufacturer_groups(results.same_manufacturer_different_products, "related-product")
        else:
            st.info("No related products from the same manufacturer found.")

    with tab3:
        if results.different_manufacturer_same_product:
            display_manufacturer_groups(results.different_manufacturer_same_product, "alternative-brand")
        else:
            st.info("No alternative brands for the same product type found.")

    # If no results in any category
    if not any([results.same_manufacturer_same_product, results.same_manufacturer_different_products, results.different_manufacturer_same_product]):
        st.info("No relevant products found in any category.")

def display_manufacturer_groups(manufacturer_groups, css_class=""):
    """Display manufacturer groups with sellers and products"""
    for manufacturer_group in manufacturer_groups:
        manufacturer = manufacturer_group.manufacturer or 'Unknown Manufacturer'
        
        # Display manufacturer name
        st.markdown(f"<h3 style='text-align: center; font-weight: bold;'><strong>Manufacturer: {manufacturer}</strong></h3>", unsafe_allow_html=True)
        
        for seller in manufacturer_group.sellers:
            seller_name = seller.seller
            products = seller.products
            
            # Display seller name
            st.markdown(f"#### **Seller: {seller_name}**")
            
            if products:
                # Create table data for this seller's products
                table_rows = []
                for product in products:
                    table_rows.append({
                        "Product Name": product.product_name,
                        "Product Type": product.product_type or '',
                        "Similarity": f"{product.similarity_score:.1%}",
                        "Code": product.product_code or '',
                        "Description": product.product_description or '',
                        "Specification": product.product_specification or ''
                    })
                
                if table_rows:
                    df = pd.DataFrame(table_rows)
                    
                    # Display the table
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=min(400, len(table_rows) * 35 + 50)
                    )
            else:
                st.info("No products found for this seller.")
            
            # Add some spacing between sellers
            st.markdown("---")

def display_grouped_results(results, product_name):
    """Display results in the old grouped format (backward compatibility)"""
    st.markdown("## 📊 Grouped Results")
    
    for manufacturer_group in results:
        # Handle both Pydantic models and dictionaries
        if hasattr(manufacturer_group, 'manufacturer'):
            # Pydantic model
            manufacturer = manufacturer_group.manufacturer or 'Unknown Manufacturer'
            sellers = manufacturer_group.sellers
        else:
            # Dictionary
            manufacturer = manufacturer_group.get('manufacturer', 'Unknown Manufacturer')
            sellers = manufacturer_group.get('sellers', [])
        
        # Display manufacturer name
        st.markdown(f"<h3 style='text-align: center; font-weight: bold;'><strong>Manufacturer: {manufacturer}</strong></h3>", unsafe_allow_html=True)
        
        for seller in sellers:
            # Handle both Pydantic models and dictionaries
            if hasattr(seller, 'seller'):
                # Pydantic model
                seller_name = seller.seller
                products = seller.products
            else:
                # Dictionary
                seller_name = seller.get('seller', 'Unknown Seller')
                products = seller.get('products', [])
            
            # Display seller name
            st.markdown(f"#### **Seller: {seller_name}**")
            
            if products:
                # Limit to top 5 products per seller
                top_products = products[:5]
                
                # Create table data for this seller's products
                table_rows = []
                for product in top_products:
                    # Handle both Pydantic models and dictionaries
                    if hasattr(product, 'product_name'):
                        # Pydantic model
                        table_rows.append({
                            "Product Name": product.product_name,
                            "Product Type": getattr(product, 'product_type', '') or '',
                            "Similarity": f"{product.similarity_score:.1%}",
                            "Code": product.product_code or '',
                            "Description": product.product_description or '',
                            "Specification": product.product_specification or ''
                        })
                    else:
                        # Dictionary
                        table_rows.append({
                            "Product Name": product.get('product_name', ''),
                            "Product Type": product.get('product_type', '') or '',
                            "Similarity": f"{product.get('similarity_score', 0):.1%}",
                            "Code": product.get('product_code', '') or '',
                            "Description": product.get('product_description', '') or '',
                            "Specification": product.get('product_specification', '') or ''
                        })
                
                if table_rows:
                    df = pd.DataFrame(table_rows)
                    
                    # Display the table
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=min(400, len(table_rows) * 35 + 50)
                    )
            else:
                st.info("No products found for this seller.")
            
            # Add some spacing between sellers
            st.markdown("---")

if __name__ == "__main__":
    # Check environment variables
    if not get_env_var('GEMINI_API_KEY'):
        st.error("❌ GEMINI_API_KEY not found in environment variables")
        st.stop()
    
    if not get_env_var('POSTGRESQL_URL'):
        st.error("❌ POSTGRESQL_URL not found in environment variables")
        st.stop()
    
    main()