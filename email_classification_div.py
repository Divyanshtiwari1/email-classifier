import ssl
import os
import warnings
import urllib3
import pandas as pd
from sqlalchemy import create_engine, text
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import json

# --- Constants and Configuration ---

# Polling interval in seconds (how often to check the database for new emails)
POLLING_INTERVAL = 10


# Database Connection Details
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"   
DB_PORT = "5432"
DB_NAME = "demo_database_detonator"

# --- Initial Setup (SSL, Warnings) ---
warnings.filterwarnings("ignore", category=UserWarning)
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

# --- Pydantic Models ---

class EmailClassification(BaseModel):
    """Structured output model for email classification"""
    
    category: Literal[
        "Initial Enquiry", 
        "Product/Service Information", 
        "Follow-up", 
        "Quotation Submission", 
        "Negotiation", 
        "Approval Process", 
        "Order Confirmation", 
        "Dispatch & Delivery", 
        "Post-Delivery / Feedback"
    ] = Field(
        description="The classification category for the email"
    )
    
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1 for the classification"
    )
    
    reasoning: str = Field(
        min_length=10,
        max_length=500,
        description="Brief explanation for why this category was chosen"
    )
    
    key_indicators: list[str] = Field(
        description="Key words or phrases that influenced the classification",
        max_items=5
    )
    
    @field_validator('key_indicators')
    @classmethod
    def validate_indicators(cls, v):
        if not v:
            return ["General content"]
        return [indicator.strip() for indicator in v if indicator.strip()]

# --- LangChain and LLM Setup ---

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.3,  # Lower temperature for more consistent structured output
    num_ctx=8000,
    format="json"  # Enable JSON mode for better structured output
)



# Create the Pydantic output parser
output_parser = PydanticOutputParser(pydantic_object=EmailClassification)

system_prompt = """
You are an expert email classifier. Your task is to analyze emails and provide structured classification results.

Classification Categories:
1. "Initial Enquiry" 
2. "Product/Service Information"
3. "Follow-up"
4. "Quotation Submission"
5. "Negotiation"
6. "Approval Process"
7. "Order Confirmation"
8. "Dispatch & Delivery"
9. "Post-Delivery / Feedback"

{format_instructions}

Analyze the email content carefully and provide your response in the exact JSON format specified.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Email Subject: {email_subject}\n\nEmail Body:\n{email_body}")
])

# Update the prompt with format instructions
prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())

chain = prompt | llm | output_parser

# --- Helper Functions ---

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def classify_email_with_retry(subject, body):
    """Invokes the LLM chain with retry logic and structured output."""
    try:
        # Ensure inputs are strings
        subject = subject or ""
        body = body or ""
        
        # Invoke the chain and get structured output
        classification_result = chain.invoke({
            "email_subject": subject, 
            "email_body": body
        })
        
        return classification_result
        
    except Exception as e:
        print(f"An API error occurred: {e}. Retrying...")
        # Return a fallback classification if all retries fail
        if "Retrying" not in str(e):  # Only create fallback on final failure
            return EmailClassification(
                category="Initial Enquiry",  # Default category
                confidence_score=0.1,
                reasoning="Classification failed, using default category",
                key_indicators=["Error in processing"]
            )
        raise e

# --- Enhanced Processing Logic ---

def process_new_emails(engine):
    """
    Fetches unclassified emails, classifies them with structured output, and updates the database.
    """
    # Query for rows that need classification
    query = 'SELECT id, subject, body FROM public.emailsdatalogger WHERE classification IS NULL;'
    
    try:
        unclassified_df = pd.read_sql(query, engine)
        
        if unclassified_df.empty:
            print("No new emails to classify. Waiting...")
            return

        print(f"Found {len(unclassified_df)} new emails to classify.")

        for index, row in unclassified_df.iterrows():
            email_id = row['id']
            subject = row['subject']
            body = row['body']
            
            print(f"Processing email ID: {email_id}...")
            
            try:
                # Get structured classification result
                classification_result = classify_email_with_retry(subject, body)
                
                # Extract only the category for now (since other columns don't exist)
                category = classification_result.category
                confidence = classification_result.confidence_score
                reasoning = classification_result.reasoning
                key_indicators = classification_result.key_indicators
                
                # Update only the classification column (existing column)
                update_query = text(
                    """
                    UPDATE public.emailsdatalogger
                    SET classification = :category
                    WHERE id = :email_id;
                    """
                )
                
                with engine.connect() as connection:
                    connection.execute(update_query, {
                        "category": category,
                        "email_id": email_id
                    })
                    connection.commit()
                    
                print(f"Successfully classified email ID {email_id}:")
                print(f"  Category: {category}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Reasoning: {reasoning}")
                print(f"  Key Indicators: {key_indicators}")
                print("-" * 50)
                
                time.sleep(1)  # Small delay between processing emails
                
            except Exception as e:
                print(f"Failed to classify email ID {email_id} after multiple retries: {e}")
    
    except Exception as e:
        print(f"An error occurred while fetching or processing emails: {e}")


# --- Database Schema Helper Function ---

def create_additional_columns(engine):
    """
    Helper function to add additional columns to store structured classification data.
    Run this once to update your database schema.
    """
    alter_queries = [
        "ALTER TABLE public.emailsdatalogger ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(3,2);",
        "ALTER TABLE public.emailsdatalogger ADD COLUMN IF NOT EXISTS classification_reasoning TEXT;",
        "ALTER TABLE public.emailsdatalogger ADD COLUMN IF NOT EXISTS key_indicators JSONB;"
    ]
    
    try:
        with engine.connect() as connection:
            for query in alter_queries:
                connection.execute(text(query))
            connection.commit()
        print("Database schema updated successfully.")
    except Exception as e:
        print(f"Error updating database schema: {e}")

# --- Service Entry Point ---

if __name__ == "__main__":
    print("Starting the enhanced email classification service with Pydantic...")
    
    try:
        # Database connection setup
        from urllib.parse import quote_plus
        encoded_password = quote_plus(DB_PASSWORD)
        db_url = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        
        # Test database connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connection established successfully.")
        
        # Uncomment the next line to update database schema (run once)
        # create_additional_columns(engine)
        
    except Exception as e:
        print(f"FATAL: Could not connect to the database: {e}")
        exit()

    # Main service loop
    try:
        while True:
            process_new_emails(engine)
            time.sleep(POLLING_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nService stopped by user.")
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
    finally:
        engine.dispose()
        print("Database connection closed.")
