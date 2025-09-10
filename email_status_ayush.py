import ssl
import os
import warnings
import urllib3
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
import time
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from datetime import datetime
import re
import json
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_status_tracker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---

# Polling interval in seconds (how often to check the database for new emails)
POLLING_INTERVAL = 15

# Database Connection Details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "demo_database_detonator")

# Updated table name
TABLE_NAME = "emailsdatalogger"

# Expected status values
EXPECTED_STATUSES = ["pending", "completed"]

# --- Initial Setup (SSL, Warnings) ---
warnings.filterwarnings("ignore", category=UserWarning)
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

# --- Pydantic Models ---

class EmailStatusAnalysis(BaseModel):
    """Structured output model for email status analysis"""
    
    status: Literal["pending", "completed"] = Field(
        description="Current status of the email thread"
    )
    
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1 for the status determination"
    )
    
    reasoning: str = Field(
        min_length=10,
        max_length=500,
        description="Brief explanation for why this status was determined"
    )
    
    key_indicators: List[str] = Field(
        description="Key words, phrases, or patterns that influenced the status decision",
        max_length=5
    )
    
    thread_summary: str = Field(
        min_length=10,
        max_length=200,
        description="Brief summary of what the email thread is about"
    )
    
    completion_signals: List[str] = Field(
        default=[],
        description="Specific indicators that suggest completion (if any)",
        max_length=3
    )
    
    pending_signals: List[str] = Field(
        default=[],
        description="Specific indicators that suggest the thread is still pending (if any)",
        max_length=3
    )
    
    @field_validator('key_indicators')
    @classmethod
    def validate_indicators(cls, v):
        if not v:
            return ["General analysis"]
        return [indicator.strip() for indicator in v if indicator.strip()]
    
    @field_validator('completion_signals', 'pending_signals')
    @classmethod
    def validate_signals(cls, v):
        return [signal.strip() for signal in v if signal.strip()]

# --- LangChain and LLM Setup ---

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser

try:
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0.3,  # Lower temperature for more consistent structured output
        num_ctx=8000,
        format="json"  # Enable JSON mode for better structured output
    )
    logger.info("LLM model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM model: {e}")
    raise

# Create the Pydantic output parser
output_parser = PydanticOutputParser(pydantic_object=EmailStatusAnalysis)

system_prompt = """
You are an expert email thread status analyzer. Your task is to analyze complete email threads and determine the current status of the conversation or business process.

Status Definitions:
- "pending": The conversation/process is still ongoing, waiting for responses, actions, or decisions
- "completed": The conversation/process appears to be finished, resolved, or concluded

Classification Guidelines:

COMPLETED Indicators:
- Confirmations received (order confirmations, delivery confirmations)
- Thank you messages indicating satisfaction
- Final approvals or sign-offs
- Delivery notifications with acknowledgment
- Payment confirmations
- Project completion statements
- "Case closed" or similar finality expressions
- Feedback provided after service completion

PENDING Indicators:
- Unanswered questions
- Requests awaiting responses
- Follow-ups needed
- Ongoing negotiations
- Quotes submitted but not yet accepted/rejected
- Tasks assigned but not completed
- Information requests not yet fulfilled
- Discussions in progress

Analysis Process:
1. Read the entire email thread chronologically
2. Identify the business process or conversation topic
3. Look for completion or pending signals
4. Consider the latest emails as most indicative of current status
5. Assess confidence based on clarity of indicators

{format_instructions}

Provide your analysis in the exact JSON format specified above.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Email Thread Analysis:\n\nSubject: {subject}\n\nEmail Thread Details:\n{email_thread}")
])

# Update the prompt with format instructions
prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())

chain = prompt | llm | output_parser

# --- Helper Functions ---

def clean_html(html_text):
    """Clean HTML tags from email body text (kept for backward compatibility)."""
    # Since body column now contains plain text, this function is no longer needed
    # but keeping it in case there are any legacy HTML content or future use
    if not isinstance(html_text, str):
        return ""
    try:
        return BeautifulSoup(html_text, "html.parser").get_text().strip()
    except Exception as e:
        logger.warning(f"HTML cleaning failed: {e}")
        return str(html_text)

def normalize_subject(subject):
    """Normalize email subject for better matching by removing common prefixes."""
    if not subject:
        return ""
    
    # Remove common email prefixes and normalize
    subject = str(subject).strip()
    
    # Remove multiple RE:, FW:, FWD: prefixes (case insensitive) - can appear multiple times
    # This regex will match any combination like "RE: RE: FW:" etc.
    prefixes_pattern = r'^(?:re:|fw:|fwd:|forward:|reply:|\s)*'
    subject = re.sub(prefixes_pattern, '', subject, flags=re.IGNORECASE)
    
    # Remove extra whitespace and normalize
    subject = ' '.join(subject.split())
    
    return subject.strip()

def extract_status_from_response(response_text):
    """Extract status from LLM response, handling various response formats."""
    if not response_text:
        return "pending"
    
    # Clean the response
    response_text = response_text.lower().strip()
    
    # Remove quotes if present
    response_text = response_text.strip('"\'')
    
    # Method 1: Check if response starts with a valid status
    if response_text.startswith('completed'):
        return "completed"
    elif response_text.startswith('pending'):
        return "pending"
    
    # Method 2: Look for status words in the response
    if 'completed' in response_text and 'pending' not in response_text:
        return "completed"
    elif 'pending' in response_text and 'completed' not in response_text:
        return "pending"
    
    # Method 3: Use regex to find status words
    import re
    status_match = re.search(r'\b(completed|pending)\b', response_text)
    if status_match:
        return status_match.group(1)
    
    # Method 4: Check for completion indicators in the explanation
    completion_indicators = [
        'finished', 'resolved', 'concluded', 'delivered', 
        'confirmed', 'approved', 'thank you', 'successful',
        'arrived', 'processed', 'payment', 'received'
    ]
    
    pending_indicators = [
        'ongoing', 'waiting', 'follow-up', 'question', 
        'request', 'inquiry', 'quotation', 'proposal'
    ]
    
    completion_count = sum(1 for indicator in completion_indicators if indicator in response_text)
    pending_count = sum(1 for indicator in pending_indicators if indicator in response_text)
    
    if completion_count > pending_count:
        return "completed"
    
    # Default to pending if unclear
    return "pending"

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def determine_email_status_with_retry(subject, email_thread):
    """Invokes the LLM chain to determine email status with retry logic and structured output."""
    try:
        subject = str(subject or "").strip()
        
        if not subject and not email_thread:
            logger.warning("Empty subject and thread - returning default status")
            return EmailStatusAnalysis(
                status="pending",
                confidence_score=0.1,
                reasoning="No content available for analysis",
                key_indicators=["Empty content"],
                thread_summary="No thread content available",
                completion_signals=[],
                pending_signals=["No content"]
            )
        
        # Invoke the chain and get structured output
        status_analysis = chain.invoke({
            "subject": subject, 
            "email_thread": email_thread
        })
        
        logger.info(f"Structured status analysis completed: {status_analysis.status} (confidence: {status_analysis.confidence_score:.2f})")
        return status_analysis
        
    except Exception as e:
        logger.error(f"Status determination API error: {e}")
        # Return a fallback structured response if all retries fail
        if "Retrying" not in str(e):  # Only create fallback on final failure
            return EmailStatusAnalysis(
                status="pending",
                confidence_score=0.1,
                reasoning="Analysis failed, defaulting to pending status",
                key_indicators=["Error in processing"],
                thread_summary="Unable to analyze thread",
                completion_signals=[],
                pending_signals=["Processing error"]
            )
        raise e

def get_complete_email_thread(engine, target_subject, target_id):
    """Get ALL emails with the same normalized subject (including those with existing status)."""
    try:
        normalized_subject = normalize_subject(target_subject)
        if not normalized_subject:
            logger.warning(f"Empty normalized subject for email ID {target_id}")
            return []

        logger.debug(f"Searching for emails with normalized subject: '{normalized_subject}'")

        # Instead of creating multiple patterns, normalize all subjects in the database
        # and compare with the normalized target subject
        query = text(f"""
            SELECT id, subject, mail_from, cc, bcc, received_time, body, classification, status
            FROM public.{TABLE_NAME}
            WHERE TRIM(
                REGEXP_REPLACE(
                    LOWER(subject), 
                    '^(re:|fw:|fwd:|forward:|reply:|\\s)*', 
                    '', 
                    'gi'
                )
            ) = LOWER(TRIM(:normalized_subject))
            AND subject IS NOT NULL
            AND TRIM(subject) != ''
            ORDER BY received_time ASC;
        """)
        
        logger.debug(f"Searching for normalized subject: '{normalized_subject}'")
        
        with engine.connect() as connection:
            result = connection.execute(query, {
                "normalized_subject": normalized_subject
            })
            thread_emails = result.fetchall()
            
        logger.info(f"Found {len(thread_emails)} emails in complete thread for subject: '{normalized_subject}'")
        
        # If no emails found with PostgreSQL regex, fall back to Python normalization
        if not thread_emails:
            logger.warning(f"No emails found with regex approach. Trying fallback method for: '{normalized_subject}'")
            
            # Fallback: Get all emails and filter in Python
            fallback_query = text(f"""
                SELECT id, subject, mail_from, cc, bcc, received_time, body, classification, status
                FROM public.{TABLE_NAME}
                WHERE subject IS NOT NULL 
                AND TRIM(subject) != ''
                ORDER BY received_time ASC;
            """)
            
            with engine.connect() as connection:
                result = connection.execute(fallback_query)
                all_emails = result.fetchall()
                
            # Filter emails with matching normalized subjects
            thread_emails = []
            for email in all_emails:
                email_subject = email[1]  # subject is at index 1
                if normalize_subject(email_subject).lower() == normalized_subject.lower():
                    thread_emails.append(email)
                    
            logger.info(f"Fallback method found {len(thread_emails)} emails for subject: '{normalized_subject}'")
        
        # Log the subjects found for debugging
        if thread_emails:
            subjects_found = [email[1] for email in thread_emails]
            logger.debug(f"Thread subjects found: {subjects_found}")
        
        # Log current status distribution in thread
        status_counts = {}
        for email in thread_emails:
            status = email[8] if len(email) > 8 and email[8] else "null"
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"Thread status distribution: {status_counts}")
        return thread_emails
        
    except Exception as e:
        logger.error(f"Error retrieving complete email thread: {e}")
        # If there's an error with the regex approach, try the simple fallback
        logger.info("Attempting simple subject matching as fallback...")
        try:
            simple_query = text(f"""
                SELECT id, subject, mail_from, cc, bcc, received_time, body, classification, status
                FROM public.{TABLE_NAME}
                WHERE LOWER(subject) LIKE LOWER(:pattern)
                ORDER BY received_time ASC;
            """)
            
            pattern = f"%{normalized_subject.lower()}%"
            
            with engine.connect() as connection:
                result = connection.execute(simple_query, {"pattern": pattern})
                thread_emails = result.fetchall()
                
            logger.info(f"Simple fallback found {len(thread_emails)} emails")
            return thread_emails
            
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {fallback_error}")
            return []

def format_email_thread(thread_emails):
    """Format the email thread for LLM analysis."""
    if not thread_emails:
        return "No related emails found."
    
    formatted_thread = []
    
    for i, email in enumerate(thread_emails, 1):
        email_id = email[0]
        subject = email[1] or ""
        mail_from = email[2] or ""  # Updated column name
        cc = email[3] or ""
        bcc = email[4] or ""
        received_time = email[5]
        body = email[6] or ""  # Updated column name (full body instead of body_preview)
        classification = email[7] or ""
        # Handle both old format (8 columns) and new format (9 columns with status)
        status = email[8] if len(email) > 8 else "unknown"
        
        # Since body now contains plain text (not HTML), no need to clean HTML
        # Just ensure it's a string and handle None values
        clean_body = str(body) if body else ""
        
        email_info = f"""
Email {i} (ID: {email_id}):
- From: {mail_from}
- CC: {cc}
- BCC: {bcc}
- Time: {received_time}
- Classification: {classification}
- Current Status: {status}
- Content: {clean_body[:500]}{"..." if len(clean_body) > 500 else ""}
"""
        formatted_thread.append(email_info.strip())
    
    return "\n\n".join(formatted_thread)

def verify_thread_status_consistency(engine, normalized_subject, expected_status):
    """Verify that all emails in a thread have the correct status after update."""
    try:
        # Use the same normalization logic as in get_complete_email_thread
        query = text(f"""
            SELECT id, subject, status 
            FROM public.{TABLE_NAME}
            WHERE TRIM(
                REGEXP_REPLACE(
                    LOWER(subject), 
                    '^(re:|fw:|fwd:|forward:|reply:|\\s)*', 
                    '', 
                    'gi'
                )
            ) = LOWER(TRIM(:normalized_subject))
            AND subject IS NOT NULL
            ORDER BY id ASC;
        """)
        
        with engine.connect() as connection:
            result = connection.execute(query, {
                "normalized_subject": normalized_subject
            })
            thread_emails = result.fetchall()
        
        # Check if all emails have the expected status
        inconsistent_emails = []
        for email in thread_emails:
            email_id, subject, status = email[0], email[1], email[2]
            if status != expected_status:
                inconsistent_emails.append(f"ID {email_id}: '{status}' (subject: '{subject}')")
        
        if inconsistent_emails:
            logger.warning(f"Thread '{normalized_subject}' has inconsistent statuses: {', '.join(inconsistent_emails)}")
            return False
        else:
            logger.info(f"[SUCCESS] Thread '{normalized_subject}' - all {len(thread_emails)} emails have status '{expected_status}'")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying thread consistency: {e}")
        
        # Fallback verification using Python normalization
        try:
            query = text(f"""
                SELECT id, subject, status 
                FROM public.{TABLE_NAME}
                WHERE subject IS NOT NULL
                ORDER BY id ASC;
            """)
            
            with engine.connect() as connection:
                result = connection.execute(query)
                all_emails = result.fetchall()
            
            # Filter and check status consistency
            thread_emails = []
            for email in all_emails:
                if normalize_subject(email[1]).lower() == normalized_subject.lower():
                    thread_emails.append(email)
            
            inconsistent_emails = []
            for email in thread_emails:
                email_id, subject, status = email[0], email[1], email[2]
                if status != expected_status:
                    inconsistent_emails.append(f"ID {email_id}: '{status}' (subject: '{subject}')")
            
            if inconsistent_emails:
                logger.warning(f"Thread '{normalized_subject}' has inconsistent statuses: {', '.join(inconsistent_emails)}")
                return False
            else:
                logger.info(f"[SUCCESS] Thread '{normalized_subject}' - all {len(thread_emails)} emails have status '{expected_status}'")
                return True
                
        except Exception as fallback_error:
            logger.error(f"Fallback verification also failed: {fallback_error}")
            return False

def update_email_status(engine, email_id, status, error_msg=None):
    """Update a single email's status in the database."""
    try:
        # The emailsdatalogger table already has status_updated_at and status_error columns
        update_query = text(f"""
            UPDATE public.{TABLE_NAME}
            SET 
                status = :status,
                status_updated_at = :timestamp,
                status_error = :error_msg
            WHERE id = :email_id;
        """)
        
        params = {
            "status": status, 
            "email_id": email_id,
            "timestamp": datetime.now(),
            "error_msg": error_msg
        }
        
        with engine.connect() as connection:
            result = connection.execute(update_query, params)
            connection.commit()
            
            if result.rowcount == 0:
                logger.warning(f"No rows updated for email ID {email_id}")
                return False
            return True
            
    except Exception as e:
        logger.error(f"Database status update failed for email ID {email_id}: {e}")
        return False

# --- Main Processing Logic ---

def process_emails_without_status(engine):
    """
    Fetches emails without status, groups them by subject thread, analyzes each thread 
    with structured Pydantic output, and updates all emails in the thread with the current status.
    """
    try:
        # Query for emails without status, ordered by id
        query = text(f"""
            SELECT id, subject, mail_from, cc, bcc, received_time, body, classification
            FROM public.{TABLE_NAME}
            WHERE status IS NULL 
            AND classification IS NOT NULL
            ORDER BY id ASC;
        """)
        
        with engine.connect() as connection:
            result = connection.execute(query)
            emails_without_status = result.fetchall()
        
        if not emails_without_status:
            logger.debug("No emails without status found")
            return 0

        logger.info(f"Found {len(emails_without_status)} emails without status to process")
        
        # Group emails by normalized subject
        subject_groups = {}
        for email_row in emails_without_status:
            email_id = email_row[0]
            subject = email_row[1]
            normalized_subject = normalize_subject(subject)
            
            if normalized_subject not in subject_groups:
                subject_groups[normalized_subject] = []
            subject_groups[normalized_subject].append(email_row)
        
        logger.info(f"Grouped emails into {len(subject_groups)} unique subject threads")
        
        total_processed = 0
        successful_threads = 0
        
        # Process each subject group
        for normalized_subject, email_group in subject_groups.items():
            logger.info(f"Processing thread: '{normalized_subject}' with {len(email_group)} emails needing status")
            
            try:
                # Use the first email's details for thread analysis
                representative_email = email_group[0]
                original_subject = representative_email[1]
                email_id_for_reference = representative_email[0]
                
                # Get ALL emails in this thread (including those with existing status)
                thread_emails = get_complete_email_thread(engine, original_subject, email_id_for_reference)
                
                # If no thread emails found, create a single-email thread for processing
                if not thread_emails:
                    logger.warning(f"No thread emails found, processing single email ID {email_id_for_reference}")
                    # Create a mock thread with just the current email data
                    thread_emails = [representative_email + (None,)]  # Add None for status column
                
                # Format the thread for analysis
                formatted_thread = format_email_thread(thread_emails)
                
                # Get structured status analysis using Pydantic
                status_analysis = determine_email_status_with_retry(original_subject, formatted_thread)
                
                # Extract the status for database update
                current_thread_status = status_analysis.status
                logger.info("-------------------------------------------------------------------")
                logger.info(f"Thread analysis result for '{normalized_subject}':")
                logger.info(f"  Status: {current_thread_status}")
                logger.info(f"  Confidence: {status_analysis.confidence_score:.2f}")
                logger.info(f"  Summary: {status_analysis.thread_summary}")
                logger.info(f"  Reasoning: {status_analysis.reasoning}")
                logger.info(f"  Key Indicators: {status_analysis.key_indicators}")
                if status_analysis.completion_signals:
                    logger.info(f"  Completion Signals: {status_analysis.completion_signals}")
                if status_analysis.pending_signals:
                    logger.info(f"  Pending Signals: {status_analysis.pending_signals}")
                logger.info("-------------------------------------------------------------------")

                
                # Update ALL emails in the complete thread with the current status
                emails_updated_in_thread = 0
                emails_changed = 0
                
                for thread_email in thread_emails:
                    email_id = thread_email[0]
                    current_status = thread_email[8] if len(thread_email) > 8 else None
                    
                    # Update ALL emails in the thread with the new status
                    if update_email_status(engine, email_id, current_thread_status):
                        emails_updated_in_thread += 1
                        
                        # Count emails that were originally NULL status for main counter
                        if email_id in [e[0] for e in email_group]:
                            total_processed += 1
                        
                        # Log status changes
                        if current_status != current_thread_status:
                            emails_changed += 1
                            logger.info(f"Updated email ID {email_id} from '{current_status}' to '{current_thread_status}'")
                        else:
                            logger.info(f"Confirmed email ID {email_id} status as '{current_thread_status}'")
                    else:
                        logger.error(f"Failed to update email ID {email_id}")
                
                successful_threads += 1
                logger.info(f"Thread '{normalized_subject}' processed: {emails_updated_in_thread} total emails updated, {emails_changed} status changes")
                logger.info(f"All {len(thread_emails)} emails in thread now have status: '{current_thread_status}'")
                
                # Verify that all emails in the thread have been updated correctly
                verify_thread_status_consistency(engine, normalized_subject, current_thread_status)
                
                # Brief pause between thread analysis to avoid overwhelming the LLM
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"Thread analysis failed: {str(e)}"
                logger.error(f"Failed to analyze thread '{normalized_subject}': {error_msg}")
                
                # Update only the emails that originally had no status with error (defaulting to pending)
                for email_row in email_group:
                    email_id = email_row[0]
                    update_email_status(engine, email_id, "pending", error_msg)
                    total_processed += 1
    
        logger.info(f"Processed {successful_threads}/{len(subject_groups)} threads successfully. Total new emails processed: {total_processed}")
        return total_processed
        
    except Exception as e:
        logger.error(f"Error in process_emails_without_status: {e}")
        return 0

def test_database_connection(engine):
    """Test the database connection and table structure."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"""
                SELECT COUNT(*) as total, 
                       COUNT(status) as with_status,
                       COUNT(*) - COUNT(status) as without_status
                FROM public.{TABLE_NAME};
            """))
            row = result.fetchone()
            total, with_status, without_status = row[0], row[1], row[2]
            
            logger.info(f"Database connection successful. Total emails: {total}, "
                       f"With status: {with_status}, Without status: {without_status}")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def verify_table_structure(engine):
    """Verify that the emailsdatalogger table has all required columns."""
    try:
        with engine.connect() as connection:
            # Check which columns exist
            column_check_query = text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = '{TABLE_NAME}'
                ORDER BY column_name;
            """)
            
            result = connection.execute(column_check_query)
            existing_columns = [row[0] for row in result.fetchall()]
            logger.info(f"Existing columns in {TABLE_NAME}: {existing_columns}")
            
            # Required columns for the email status tracker
            required_columns = [
                'id', 'subject', 'mail_from', 'cc', 'bcc', 
                'received_time', 'body', 'classification', 
                'status', 'status_updated_at', 'status_error'
            ]
            
            missing_columns = [col for col in required_columns if col not in existing_columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            else:
                logger.info("All required columns are present in the table")
                return True
                    
    except Exception as e:
        logger.error(f"Error in verify_table_structure: {e}")
        return False

# Test function to verify the normalization works correctly
def test_subject_normalization():
    """Test function to verify subject normalization works correctly."""
    test_cases = [
        "Meeting Request",
        "RE: Meeting Request", 
        "Re: Meeting Request",
        "FW: Meeting Request",
        "RE: RE: Meeting Request",
        "RE: FW: Meeting Request", 
        "   RE:    Meeting Request   ",
        "Reply: Meeting Request",
        "Forward: Meeting Request",
        "FWD: RE: Meeting Request"
    ]
    
    logger.info("Testing subject normalization:")
    for subject in test_cases:
        normalized = normalize_subject(subject)
        logger.info(f"'{subject}' -> '{normalized}'")
    
    # All should normalize to "Meeting Request"
    expected = "Meeting Request"
    for subject in test_cases:
        normalized = normalize_subject(subject)
        if normalized != expected:
            logger.error(f"Failed for '{subject}': got '{normalized}', expected '{expected}'")
        else:
            logger.debug(f"✓ '{subject}' normalized correctly")
    
    logger.info("Subject normalization test completed!")

# Test function for Pydantic model
def test_pydantic_model():
    """Test the Pydantic model with sample data."""
    logger.info("Testing Pydantic EmailStatusAnalysis model...")
    
    try:
        # Test valid data
        test_analysis = EmailStatusAnalysis(
            status="completed",
            confidence_score=0.85,
            reasoning="Order has been delivered and customer confirmed receipt",
            key_indicators=["delivered", "confirmed", "thank you"],
            thread_summary="Customer order delivery and confirmation",
            completion_signals=["delivered", "confirmed"],
            pending_signals=[]
        )
        
        logger.info(f"✓ Pydantic model test successful: {test_analysis.status}")
        logger.info(f"✓ Model validation working: confidence={test_analysis.confidence_score}")
        return True
        
    except Exception as e:
        logger.error(f"Pydantic model test failed: {e}")
        return False

# --- Service Entry Point ---

if __name__ == "__main__":
    logger.info("Starting the enhanced email status tracking service with Pydantic...")
    
    # Run tests
    test_subject_normalization()
    test_pydantic_model()
    
    try:
        # Create database engine
        from urllib.parse import quote_plus
        encoded_password = quote_plus(DB_PASSWORD)
        db_url = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300)
        
        # Test connection
        if not test_database_connection(engine):
            logger.error("FATAL: Database connection failed")
            exit(1)
        
        # Verify table structure
        if not verify_table_structure(engine):
            logger.error("FATAL: Table structure verification failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"FATAL: Could not create database engine: {e}")
        exit(1)

    # Main service loop
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    try:
        while True:
            try:
                processed_count = process_emails_without_status(engine)
                consecutive_failures = 0  # Reset on success
                
                if processed_count > 0:
                    logger.info(f"Status update batch completed. Processed {processed_count} emails.")
                    
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Processing batch failed (attempt {consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({max_consecutive_failures}). Stopping service.")
                    break
                    
                # Wait longer after failures
                time.sleep(POLLING_INTERVAL * 2)
                continue
            
            time.sleep(POLLING_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        engine.dispose()
        logger.info("Database connection closed. Status tracking service terminated.")