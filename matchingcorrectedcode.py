from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field, Session, create_engine, select, text, Column
from pgvector.sqlalchemy import Vector
from typing import List
from google.genai import types
import os
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get environment variables (works both locally and on Streamlit Cloud)
def get_env_var(key):
    """Get environment variable from st.secrets (Streamlit Cloud) or os.environ (local)"""
    try:
        import streamlit as st
        return st.secrets[key]
    except:
        return os.getenv(key)

# New output schema models
class CategorizedProduct(BaseModel):
    id: Optional[int] = None
    product_name: str
    product_description: Optional[str] = None
    product_specification: Optional[str] = None
    manufacture_name: Optional[str] = None
    product_code: Optional[str] = None
    page_number: Optional[int] = None
    product_type: str
    seller_name: Optional[str] = None
    similarity_score: float

class CategorizedSeller(BaseModel):
    seller: str
    products: List[CategorizedProduct]

class CategorizedManufacturer(BaseModel):
    manufacturer: str
    sellers: List[CategorizedSeller]

class CategorizedResults(BaseModel):
    same_manufacturer_same_product: List[CategorizedManufacturer] = Field(description="Exact matches from preferred manufacturer")
    same_manufacturer_different_products: List[CategorizedManufacturer] = Field(description="Related products from preferred manufacturer")
    different_manufacturer_same_product: List[CategorizedManufacturer] = Field(description="Same product type from other manufacturers")

class ProductTypeOutput(BaseModel):
    product_type: str = Field(..., description="The standardized product type")

def join_product_context(product_name: str, description: str = None, specification: str = None) -> str:

    """
    Join product name, description, and specification into a single string.
    Only includes non-empty values.
    """
    context_parts = [f"Product Name: {product_name}"]
    if description:
        context_parts.append(f"Description: {description}")
    if specification:
        context_parts.append(f"Specification: {specification}")
    
    return "\n".join(context_parts)

def product_type_detection(context_parts):
    """
    Detect product type using LangChain with structured output
    """
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert in classifying industrial products into standardized categories. Your task is to determine the most appropriate product type based on the given product information.
    
    IMPORTANT - Product Type Guidelines:
    - Keep product_type SIMPLE and STANDARDIZED
    - Avoid specific technical details in product_type
    - Use generic categories like: "Current Monitoring Relay", "Voltage Relay", "Timer Relay", etc.
    - Choose the most general category that fits the product
    - If unsure, choose a broader category rather than a specific one
    
    Examples:
    - "Digital Current Monitoring Relay" → "Current Monitoring Relay"
    - "11 KV VCB Panel" → "Switchgear"
    - "Siemens 3RT Contactors" → "Contactor"
    - "ABB TMAX T7 Circuit Breaker" → "Circuit Breaker"
    - "Centrifugal Pump" → "Pump"
    - "Pressure Sensor" → "Sensor"
    
    Product Information:
    {context_parts}
    
    Return ONLY the product type ,nothing else.
    """)
    
    api_key = get_env_var('GEMINI_API_KEY')
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.4,
        google_api_key=api_key
    )
    chain = prompt_template | llm.with_structured_output(ProductTypeOutput)
    result = chain.invoke({"context_parts": context_parts})
    return result.product_type

def get_embedding(text: str, client) -> List[float]:
    """Generate embedding for given text using Google Gemini embedding model"""
    if not text or text.strip() == "":
        return [0.0] * 768  # Return zero vector for empty text
    
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text.strip(),
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        [embedding_obj] = result.embeddings
        return embedding_obj.values
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * 768

def get_product_embedding(product_type: str, product_name: str, client) -> list[float]:
    """Generate embedding for product by combining product_type + product_name"""
    combined_text = f"{product_type} {product_name}".strip()
    return get_embedding(combined_text, client)

def get_manufacturer_embedding(manufacture_name: str, client) -> list[float]:
    """Generate embedding for manufacturer name"""
    return get_embedding(manufacture_name, client)


POSTGRESQL_URL = get_env_var('POSTGRESQL_URL')
engine = create_engine(POSTGRESQL_URL)

# Database model
class CatalogueProductsV7(SQLModel, table=True):
    __tablename__ = "catalogue_products_v7"
    __table_args__ = (
        {'extend_existing': True, 'keep_existing': False}
    )
    id: int = Field(default=None, primary_key=True)
    product_name: str 
    product_description: Optional[str] 
    product_specification: Optional[str]
    manufacture_name: Optional[str] 
    product_code: Optional[str]
    page_number: int 
    product_type: str 
    seller_name: str
    product_embedding: list[float]= Field(sa_column=Column(Vector(768)))
    manufacturer_embedding: Optional[list[float]]= Field(default=None, sa_column=Column(Vector(768)))



def search_products_with_embeddings(
    product_type: str,
    product_name: str,
    manufacture_name: Optional[str] = None,
    manufacturer_weight: float = 0.3,
    threshold: float = 0.7,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Search products using weighted embeddings for both product and manufacturer
    
    Args:
        product_type: Product type (e.g., "Current Monitoring Relay")
        product_name: Product name (e.g., "Digital Current Monitoring Relay")
        manufacture_name: Manufacturer name (optional)
        manufacturer_weight: Weight for manufacturer embedding (0.0-1.0, default 0.3)
        threshold: Minimum similarity score (0.0-1.0, default 0.7)
        limit: Maximum number of results to return
    
    Returns:
        List of matching products with similarity scores
    """
    # Validate inputs
    if not 0.0 <= manufacturer_weight <= 1.0:
        raise ValueError("Manufacturer weight must be between 0.0 and 1.0")
    
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    # Initialize Gemini client
    api_key = get_env_var('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    client = genai.Client(api_key=api_key)
    
    # Generate product embedding (product_type + product_name)
    product_embedding = get_product_embedding(product_type, product_name, client)
    
    # Generate manufacturer embedding if provided
    manufacturer_embedding = None
    if manufacture_name and manufacture_name.strip():
        manufacturer_embedding = get_manufacturer_embedding(manufacture_name.strip(), client)
    
    with Session(engine) as session:
        if manufacturer_embedding:
            # Weighted search with both product and manufacturer embeddings
            product_weight = 1.0 - manufacturer_weight
            
            query = text(f"""
                SELECT 
                    id, product_name, product_description, product_specification,
                    manufacture_name, product_code, page_number, product_type, seller_name,
                    (
                        :product_weight * (1 - (product_embedding <=> CAST(:product_embedding AS vector))) +
                        :manufacturer_weight * (1 - (manufacturer_embedding <=> CAST(:manufacturer_embedding AS vector)))
                    ) as similarity_score
                FROM catalogue_products_v7
                WHERE manufacturer_embedding IS NOT NULL
                AND manufacture_name IS NOT NULL 
                AND manufacture_name != ''
                AND (
                    :product_weight * (1 - (product_embedding <=> CAST(:product_embedding AS vector))) +
                    :manufacturer_weight * (1 - (manufacturer_embedding <=> CAST(:manufacturer_embedding AS vector)))
                ) >= :threshold
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)
            
            result = session.exec(
                query.bindparams(
                    product_embedding=str(product_embedding),
                    manufacturer_embedding=str(manufacturer_embedding),
                    product_weight=product_weight,
                    manufacturer_weight=manufacturer_weight,
                    threshold=threshold,
                    limit=limit
                )
            )
        else:
            # Search with only product embedding (no manufacturer)
            query = text("""
                SELECT 
                    id, product_name, product_description, product_specification,
                    manufacture_name, product_code, page_number, product_type, seller_name,
                    (1 - (product_embedding <=> CAST(:product_embedding AS vector))) as similarity_score
                FROM catalogue_products_v7
                WHERE (1 - (product_embedding <=> CAST(:product_embedding AS vector))) >= :threshold
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)
            
            result = session.exec(
                query.bindparams(
                    product_embedding=str(product_embedding),
                    threshold=threshold,
                    limit=limit
                )
            )
        
        # Convert results to list of dictionaries
        products = []
        for row in result:
            products.append({
                "id": row.id,
                "product_name": row.product_name,
                "product_description": row.product_description,
                "product_specification": row.product_specification,
                "manufacture_name": row.manufacture_name,
                "product_code": row.product_code,
                "page_number": row.page_number,
                "product_type": row.product_type,
                "seller_name": row.seller_name,
                "similarity_score": float(row.similarity_score)
            })
        
        return products

def group_products_by_manufacturer(products: List[Dict[str, Any]], top_products_per_seller: int = 5) -> List[Dict[str, Any]]:
    """
    Group products by manufacturer, then by seller, with top N products per seller
    
    Args:
        products: List of product dictionaries from search results
        top_products_per_seller: Number of top products to return per seller (default 5)
    
    Returns:
        List of manufacturer groups with seller subgroups and their top products
    """
    if not products:
        return []
    
    # Group by manufacturer first
    manufacturer_groups = {}
    
    for product in products:
        manufacturer = product.get('manufacture_name', 'Unknown Manufacturer')
        seller = product.get('seller_name', 'Unknown Seller')
        
        # Initialize manufacturer group if not exists
        if manufacturer not in manufacturer_groups:
            manufacturer_groups[manufacturer] = {}
        
        # Initialize seller group if not exists
        if seller not in manufacturer_groups[manufacturer]:
            manufacturer_groups[manufacturer][seller] = []
        
        # Add product to seller group
        manufacturer_groups[manufacturer][seller].append(product)
    
    # Sort products by similarity score and get top N per seller
    result = []
    
    for manufacturer, seller_groups in manufacturer_groups.items():
        manufacturer_data = {
            'manufacturer': manufacturer,
            'sellers': []
        }
        
        for seller, seller_products in seller_groups.items():
            # Sort products by similarity score (descending) and take top N
            sorted_products = sorted(
                seller_products, 
                key=lambda x: x.get('similarity_score', 0), 
                reverse=True
            )[:top_products_per_seller]
            
            seller_data = {
                'seller': seller,
                'products': sorted_products
            }
            
            manufacturer_data['sellers'].append(seller_data)
        
        result.append(manufacturer_data)
    
    return result

def refine_search_results_with_llm(
    context_parts: str,
    detected_product_type: str,
    manufacture_name: Optional[str],
    results_json: str
) -> CategorizedResults:
    """
    Refine search results using LLM to categorize into three groups:
    1. Same manufacturer, same product (exact matches)
    2. Same manufacturer, different products (related products from preferred manufacturer)
    3. Different manufacturer, same product (same product type from other manufacturers)
    
    Args:
        context_parts: Joined product context from join_product_context
        detected_product_type: Product type from product_type_detection
        manufacture_name: Preferred manufacturer name
        results_json: JSON string of grouped results from group_products_by_manufacturer
    
    Returns:
        CategorizedResults with three categories of products
    """
    # Create prompt for LLM categorization
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert product recommendation assistant. Your task is to categorize search results into three specific groups based on relevance and manufacturer preference.

    CATEGORIZATION STRATEGY:
    1. **Same Manufacturer, Same Product**: Exact matches from the preferred manufacturer (highest priority)
       - Same product name/model from preferred manufacturer
       - Similar product names with minor variations
       - Same product type with very similar specifications
       - Products that are essentially identical to the search query
       - Allow small differences in naming conventions and abbreviations

    2. **Same Manufacturer, Different Products**: Related products from the preferred manufacturer
       - Same manufacturer but DIFFERENT product types/categories
       - Related products that serve similar functions but are not the same type
       - Products from preferred manufacturer that are alternatives in different categories
       - Only use this category if the preferred manufacturer has products in different categories

    3. **Different Manufacturer, Same Product**: Same product type from other manufacturers
       - Same product category from different manufacturers
       - Alternative brands offering similar functionality
       - Competitor products with similar specifications

    FILTERING CRITERIA:
    - Keep only the BEST 1 product per seller in each category
    - Consider manufacturer preference if specified
    - Focus on product functionality and specifications

    OUTPUT REQUIREMENTS:
    - Categorize products into the three specified groups
    - Maintain the manufacturer → seller → products hierarchy within each category
    - Return in the structured format with three categories

    Product Context: {context_parts}
    Product Type: {detected_product_type}
    Preferred Manufacturer: {manufacture_name}
    
    Current Search Results (JSON):
    {results_json}

    Please categorize these results into:
    1. same_manufacturer_same_product: Exact matches from preferred manufacturer
    2. same_manufacturer_different_products: Related products from preferred manufacturer  
    3. different_manufacturer_same_product: Same product type from other manufacturers

    Return the categorized results in the structured format with three categories.
    """)
 
    api_key = get_env_var('GEMINI_API_KEY')
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.0,
        google_api_key=api_key
    )
    llm_with_structure = llm.with_structured_output(CategorizedResults)
    chain = prompt_template | llm_with_structure
    
    # Get categorized results from LLM
    try:
        categorized_results = chain.invoke({
            "context_parts": context_parts,
            "detected_product_type": detected_product_type,
            "manufacture_name": manufacture_name or "None",
            "results_json": results_json
        })
        
        return categorized_results
        
    except Exception as e:
        print(f"Error categorizing results with LLM: {e}")
        # Fallback: return empty categorized results
        return CategorizedResults(
            same_manufacturer_same_product=[],
            same_manufacturer_different_products=[],
            different_manufacturer_same_product=[]
        )