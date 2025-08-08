from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field, Session, create_engine, select, text, Column
from pgvector.sqlalchemy import Vector
from typing import List
from google.genai import types
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from typing import Literal

load_dotenv()

class ProductTypeOutput(BaseModel):
    """Structured output for product type detection"""
    product_type: str = Field(description="The detected product type")


class Product(BaseModel):
    id: Optional[int] = None
    product_name: str
    product_description: Optional[str] = None
    product_specification: Optional[str] = None
    manufacture_name: Optional[str] = None
    product_code: Optional[str] = None
    product_type: Optional[str] = None
    seller_name: Optional[str] = None
    embedding_similarity_score: float
    llm_similarity_score: float
    category: Literal["same_manufacturer_same_product", "same_manufacturer_different_products", "different_manufacturer_same_product"]

class ExtractProductList(BaseModel):
    products: List[Product]


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
    - If unsure, choose a broader category rath
                                                       er than a specific one
    
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
    
    api_key = os.getenv('GEMINI_API_KEY')
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

POSTGRESQL_URL = os.getenv('POSTGRESQL_URL')
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
    api_key = os.getenv('GEMINI_API_KEY')
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
                    similarity_score
                FROM (
                    SELECT 
                        id, product_name, product_description, product_specification,
                        manufacture_name, product_code, page_number, product_type, seller_name,
                        (
                            :product_weight * (1 - (product_embedding <=> CAST(:product_embedding AS vector))) +
                            :manufacturer_weight * (1 - (manufacturer_embedding <=> CAST(:manufacturer_embedding AS vector)))
                        ) as similarity_score,
                        ROW_NUMBER() OVER (PARTITION BY seller_name, manufacture_name ORDER BY 
                            :product_weight * (1 - (product_embedding <=> CAST(:product_embedding AS vector))) +
                            :manufacturer_weight * (1 - (manufacturer_embedding <=> CAST(:manufacturer_embedding AS vector))) DESC
                        ) as rn
                    FROM catalogue_products_v7
                    WHERE manufacturer_embedding IS NOT NULL
                    AND manufacture_name IS NOT NULL 
                    AND manufacture_name != ''
                    AND (
                        :product_weight * (1 - (product_embedding <=> CAST(:product_embedding AS vector))) +
                        :manufacturer_weight * (1 - (manufacturer_embedding <=> CAST(:manufacturer_embedding AS vector)))
                    ) >= :threshold
                ) ranked
                WHERE rn <= 5
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
                    similarity_score
                FROM (
                    SELECT 
                        id, product_name, product_description, product_specification,
                        manufacture_name, product_code, page_number, product_type, seller_name,
                        (1 - (product_embedding <=> CAST(:product_embedding AS vector))) as similarity_score,
                        ROW_NUMBER() OVER (PARTITION BY seller_name, manufacture_name ORDER BY 
                            (1 - (product_embedding <=> CAST(:product_embedding AS vector))) DESC
                        ) as rn
                    FROM catalogue_products_v7
                    WHERE (1 - (product_embedding <=> CAST(:product_embedding AS vector))) >= :threshold
                ) ranked
                WHERE rn <= 5
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
                "embedding_similarity_score": float(row.similarity_score)
            })
        
        return products
 
def refine_search_results_with_llm(
    search_results_json: str,
    product_name: str,
    detected_product_type: str,
    description: Optional[str] = None,
    specification: Optional[str] = None,
    manufacture_name: Optional[str] = None
) -> ExtractProductList:
    """
    Refine search results using LLM to categorize into three groups and generate LLM similarity scores:
    1. Same manufacturer, same product (exact matches)
    2. Same manufacturer, different products (related products from preferred manufacturer)  
    3. Different manufacturer, same product (same product type from other manufacturers)
    
    Returns ExtractProductList with products containing:
    - embedding_similarity_score: Original vector similarity score
    - llm_similarity_score: LLM-generated similarity score (0.0-1.0)
    - category: Assigned category
    
    Args:
        search_results_json: JSON string containing search results from search_products_with_embeddings
        product_name: Product name from user query
        detected_product_type: Already detected product type (no need to detect again)
        description: Product description from user query (optional)
        specification: Product specification from user query (optional)
        manufacture_name: Manufacturer name from user query (optional)
    """
    import json
    from collections import defaultdict
    
    # Step 1: Prepare context (no need to detect product type again)
    context_parts = join_product_context(product_name, description, specification)

    # Step 2: Parse search results from JSON
    try:
        products = json.loads(search_results_json)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for search results")

    # Step 3: Group products by seller name, then by manufacturer within each seller
    grouped_by_seller = defaultdict(lambda: defaultdict(list))
    for product in products:
        seller = product.get('seller_name', 'Unknown')
        if not seller or seller.strip() == '':
            seller = 'Unknown'
        
        manufacturer = product.get('manufacture_name', 'Unknown')
        if not manufacturer or manufacturer.strip() == '':
            manufacturer = 'Unknown'
            
        grouped_by_seller[seller][manufacturer].append(product)
    
    # Step 3.5: Remove the filtering step since it's now done at database level
    # Convert nested structure to regular dict for JSON serialization
    nested_grouped_dict = {}
    for seller, manufacturers in grouped_by_seller.items():
        nested_grouped_dict[seller] = dict(manufacturers)
    
    # Convert grouped results to JSON for LLM
    grouped_results_json = json.dumps(nested_grouped_dict, indent=2)

    # Step 4: Initialize LLM for categorization
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key
    )

    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert in categorizing product search results. The search results are NESTED: SELLER → MANUFACTURER → PRODUCTS.
    
    STRUCTURE EXPLANATION:
    - Each SELLER has multiple MANUFACTURERS
    - Each MANUFACTURER (within a seller) has multiple PRODUCTS
    - You need to pick ONLY 1 BEST PRODUCT PER MANUFACTURER (within each seller), and ONLY IF IT'S RELEVANT

    For EACH SELLER → EACH MANUFACTURER, analyze their products and assign the BEST ONE to ONE of these categories (use STRICT matching logic):
                                                                                     
    1. **same_manufacturer_same_product**:
    - Product is from the **preferred manufacturer**                                                 
    - Functionally equivalent to the user's intended product
    - Example: "Control Relay" and "Monitoring Relay" = SAME category if both are relays
    - Match based on **core purpose**, **description**, and **specifications**, not just exact name

    2. **same_manufacturer_different_products**:
    - Also from the **preferred manufacturer**                                                     
    - Not functionally identical but clearly **related** or **complementary**
    - Example: If query is "Motor", this could include "Motor Protection Relay" or "Starter"
    - DO NOT include vague or unrelated products


    3. **different_manufacturer_same_product**:
    - Skip irrelevant or vague products from other manufacturers - only categorize products that serve the EXACT SAME function as the user query
    - Must be functionally IDENTICAL and serve the same purpose
    - But from a manufacturer OTHER THAN the preferred one
    - Match based on **identical function**, not just similar name
    - Example: "Current Monitor" from Siemens = "Current Monitoring Relay" from ABB

    STRICT FILTERING RULES:
    - For each SELLER → each MANUFACTURER: pick ONLY 1 BEST PRODUCT (per manufacturer)
    - If manufacturer has NO RELEVANT products for the seller, SKIP that manufacturer
    - first check if manufacturer is present in User Query Context: {context_parts} ,Product Type: {detected_product_type} ,Preferred Manufacturer: {manufacture_name} if not then it then if its relevant it will fall under **different_manufacturer_same_product**:
    - Functional similarity
    - Use case compatibility
    - Product name, description, specifications

    - Pick the **single most relevant** product per seller-manufacturer pair
    - MAXIMUM 1 product per manufacturer per seller
    - AVOID vague, generic, or unrelated products – return NOTHING if unsure

    SCORING:(be strict in your evaluation)
    - Generate `llm_similarity_score` (between 0.0 and 1.0) based on:
    - Functional similarity to user query (most important)                                                   
    - Description relevance
    - Specification compatibility
    - Product name
    - Application match

    User Query Context: {context_parts}
    Product Type: {detected_product_type}
    Preferred Manufacturer: {manufacture_name}

    Search Results (Nested: Seller → Manufacturer → Products):
    {grouped_results}

    Return ExtractProductList with each product having:
    - embedding_similarity_score (retain original value)
    - llm_similarity_score (you generate based on relevance)
    - category (choose from: "same_manufacturer_same_product", "same_manufacturer_different_products", "different_manufacturer_same_product")
    - MAXIMUM 1 product per manufacturer per seller
    """)

    llm_with_structure = llm.with_structured_output(ExtractProductList)
    chain = prompt_template | llm_with_structure

    # Step 5: Get categorized results using grouped data
    categorized_results = chain.invoke({
        "context_parts": context_parts,
        "detected_product_type": detected_product_type,
        "manufacture_name": manufacture_name or "None",
        "grouped_results": grouped_results_json
    })

    return categorized_results