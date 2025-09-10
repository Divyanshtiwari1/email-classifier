import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import warnings
from bs4 import BeautifulSoup
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Email Classification Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection configuration
@st.cache_resource
def init_database_connection():
    """Initialize database connection"""
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "demo_database_detonator"
    
    try:
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Data loading functions
@st.cache_data(ttl=30)  # Cache for 30 seconds to allow for real-time updates
def load_email_data(_engine, start_date=None, end_date=None):
    """Load email data from database with optional date filtering"""
    if _engine is None:
        return pd.DataFrame()
    
    base_query = '''
    SELECT 
        id,
        subject,
        mail_from as "from",
        cc,
        bcc,
        received_time,
        body as body_preview,
        hasattachment as attachment,
        classification,
        status,
        status_updated_at,
        status_error,
        DATE(received_time) as email_date
    FROM public.emailsdatalogger
    WHERE classification IS NOT NULL
    '''
    
    # Add date filtering if provided
    if start_date and end_date:
        base_query += f" AND DATE(received_time) BETWEEN '{start_date}' AND '{end_date}'"
    
    base_query += ' ORDER BY received_time DESC'
    
    try:
        df = pd.read_sql(base_query, _engine)
        if not df.empty:
            # Handle "received_time" column (timestamp without time zone)
            df['received_time'] = pd.to_datetime(df['received_time'], errors='coerce')
            
            # Handle email_date column
            df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce')
            
            # Convert status_updated_at to datetime if it exists (timestamp without time zone)
            if 'status_updated_at' in df.columns:
                df['status_updated_at'] = pd.to_datetime(df['status_updated_at'], errors='coerce')
            
            # Convert hasattachment boolean to string for display
            if 'attachment' in df.columns:
                df['attachment'] = df['attachment'].apply(lambda x: 'Yes' if x else '')
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def normalize_subject(subject):
    """
    Normalize email subject by removing 'RE:', 'FW:', 'FWD:', etc. and extra whitespace
    to group related emails together
    """
    if not subject or pd.isna(subject):
        return "No Subject"
    
    # Convert to string and strip whitespace
    normalized = str(subject).strip()
    
    # Remove common email prefixes (case-insensitive)
    prefixes = [r'^RE:\s*', r'^FW:\s*', r'^FWD:\s*', r'^FORWARD:\s*', r'^AW:\s*', r'^WG:\s*']
    
    for prefix in prefixes:
        normalized = re.sub(prefix, '', normalized, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Handle empty subjects after normalization
    if not normalized:
        return "No Subject"
    
    return normalized

def calculate_email_thread_completion_time(df):
    """
    Calculate completion time for email threads by grouping emails with same normalized subject
    and finding the time difference between first and last email in the thread
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original dataframe
    thread_df = df.copy()
    
    # Filter out emails with invalid received_time
    thread_df = thread_df[thread_df['received_time'].notna()]
    
    if thread_df.empty:
        return pd.DataFrame()
    
    # Add normalized subject column
    thread_df['normalized_subject'] = thread_df['subject'].apply(normalize_subject)
    
    # Group by normalized subject and calculate thread statistics
    thread_stats = []
    
    for normalized_subject, group in thread_df.groupby('normalized_subject'):
        if len(group) > 1:  # Only consider threads with multiple emails
            # Sort by received time and filter out NaT values
            group_sorted = group.sort_values('received_time').dropna(subset=['received_time'])
            
            if len(group_sorted) < 2:  # Skip if not enough valid timestamps
                continue
            
            first_email = group_sorted.iloc[0]
            last_email = group_sorted.iloc[-1]
            
            # Check for valid timestamps
            if pd.isna(first_email['received_time']) or pd.isna(last_email['received_time']):
                continue
            
            # Calculate completion time in days
            time_diff = last_email['received_time'] - first_email['received_time']
            completion_days = time_diff.total_seconds() / (24 * 3600)  # Convert to days
            
            # Check if thread has reply emails (emails with RE:, FW:, etc. in original subject)
            reply_count = len(group[group['subject'].str.contains(r'^(RE|FW|FWD):', case=False, na=False)])
            
            thread_stats.append({
                'normalized_subject': normalized_subject,
                'original_subject': first_email['subject'] if first_email['subject'] else 'No Subject',
                'thread_id': f"thread_{hash(normalized_subject) % 100000}",
                'email_count': len(group),
                'reply_count': reply_count,
                'first_email_date': first_email['received_time'],
                'last_email_date': last_email['received_time'],
                'completion_days': round(completion_days, 2) if completion_days >= 0 else 0,
                'completion_hours': round(completion_days * 24, 2) if completion_days >= 0 else 0,
                'first_sender': first_email['from'] if first_email['from'] else 'Unknown',
                'last_sender': last_email['from'] if last_email['from'] else 'Unknown',
                'classification': first_email['classification'] if first_email['classification'] else 'Unknown',
                'status': last_email['status'] if 'status' in last_email and last_email['status'] else None,
                'emails_in_thread': group['id'].tolist()
            })
    
    return pd.DataFrame(thread_stats)

def show_task_completion_analysis(df):
    """Show task completion time analysis"""
    st.header("⏱️ Task Completion Time Analysis")
    
    if df.empty:
        st.warning("No email data available for analysis.")
        return
    
    # Calculate thread completion times
    thread_df = calculate_email_thread_completion_time(df)
    
    if thread_df.empty:
        st.warning("No email threads with multiple messages found. Task completion analysis requires email conversations with replies.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Email Threads", len(thread_df))
    
    with col2:
        avg_completion = thread_df['completion_days'].mean()
        st.metric("Avg Completion Time", f"{avg_completion:.1f} days")
    
    with col3:
        fastest_completion = thread_df['completion_days'].min()
        st.metric("Fastest Completion", f"{fastest_completion:.1f} days")
    
    with col4:
        longest_completion = thread_df['completion_days'].max()
        st.metric("Longest Completion", f"{longest_completion:.1f} days")
    
    st.markdown("---")
    
    # Completion time distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Completion Time Distribution")
        
        # Histogram of completion times
        fig_hist = px.histogram(
            thread_df,
            x='completion_days',
            nbins=20,
            title="Distribution of Task Completion Times",
            labels={'completion_days': 'Completion Time (Days)', 'count': 'Number of Threads'},
            color_discrete_sequence=['skyblue']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Completion Time by Category")
        
        # Box plot by category
        fig_box = px.box(
            thread_df,
            x='classification',
            y='completion_days',
            title="Completion Time by Email Category",
            labels={'completion_days': 'Completion Time (Days)', 'classification': 'Category'}
        )
        fig_box.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Detailed thread analysis table
    st.subheader("📋 Email Thread Analysis")
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Email threads with completion time analysis:**")
    with col2:
        sort_option = st.selectbox(
            "Sort by:",
            ["Longest First", "Shortest First", "Most Recent", "Most Emails"],
            key="sort_threads"
        )
    
    # Apply sorting
    if sort_option == "Longest First":
        display_df = thread_df.sort_values('completion_days', ascending=False)
    elif sort_option == "Shortest First":
        display_df = thread_df.sort_values('completion_days', ascending=True)
    elif sort_option == "Most Recent":
        display_df = thread_df.sort_values('last_email_date', ascending=False)
    else:  # Most Emails
        display_df = thread_df.sort_values('email_count', ascending=False)
    
    # Display thread analysis table
    for idx, thread in display_df.iterrows():
        with st.container():
            st.markdown("---")
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**📧 {thread['original_subject']}**")
                st.markdown(f"**Category:** {thread['classification']}")
                st.markdown(f"**Thread:** {thread['email_count']} emails ({thread['reply_count']} replies)")
                
                # Safe datetime formatting with null checks
                try:
                    if pd.notna(thread['first_email_date']):
                        first_date_str = thread['first_email_date'].strftime('%Y-%m-%d %H:%M')
                    else:
                        first_date_str = "Unknown date"
                    
                    if pd.notna(thread['last_email_date']):
                        last_date_str = thread['last_email_date'].strftime('%Y-%m-%d %H:%M')
                    else:
                        last_date_str = "Unknown date"
                    
                    st.markdown(f"**Started:** {first_date_str} by {thread['first_sender']}")
                    st.markdown(f"**Completed:** {last_date_str} by {thread['last_sender']}")
                    
                except (AttributeError, TypeError, ValueError):
                    st.markdown(f"**Started:** Invalid date by {thread['first_sender']}")
                    st.markdown(f"**Completed:** Invalid date by {thread['last_sender']}")
            
            with col2:
                # Completion time with color coding
                days = thread['completion_days']
                if days <= 1:
                    st.success(f"⚡ {days} days\n({thread['completion_hours']:.1f} hours)")
                elif days <= 7:
                    st.info(f"📅 {days} days")
                elif days <= 30:
                    st.warning(f"📆 {days} days")
                else:
                    st.error(f"⏳ {days} days")
                
                # Status if available
                if thread['status']:
                    status_display, status_color = get_status_display(thread['status'])
                    st.markdown(f"**Status:** {status_display}")
            
            with col3:
                if st.button(f"👀 View Thread", key=f"thread_{thread['thread_id']}", use_container_width=True):
                    # Show thread details in expander
                    with st.expander(f"Thread Details: {thread['original_subject']}", expanded=True):
                        thread_emails = df[df['id'].isin(thread['emails_in_thread'])].sort_values('received_time')
                        
                        for i, (_, email) in enumerate(thread_emails.iterrows()):
                            # Safe datetime handling for thread details
                            try:
                                if pd.notna(email['received_time']):
                                    email_date_str = email['received_time'].strftime('%Y-%m-%d %H:%M')
                                else:
                                    email_date_str = "Unknown date"
                            except (AttributeError, TypeError, ValueError):
                                email_date_str = "Invalid date"
                            
                            st.markdown(f"**Email {i+1}:**")
                            st.markdown(f"- **From:** {email['from'] if email['from'] else 'Unknown'}")
                            st.markdown(f"- **Subject:** {email['subject'] if email['subject'] else 'No Subject'}")
                            st.markdown(f"- **Date:** {email_date_str}")
                            
                            # Show time difference from previous email
                            if i > 0:
                                prev_email = thread_emails.iloc[i-1]
                                if pd.notna(email['received_time']) and pd.notna(prev_email['received_time']):
                                    time_diff = email['received_time'] - prev_email['received_time']
                                    hours_diff = time_diff.total_seconds() / 3600
                                    if hours_diff < 24:
                                        st.markdown(f"- **Response Time:** {hours_diff:.1f} hours")
                                    else:
                                        st.markdown(f"- **Response Time:** {hours_diff/24:.1f} days")
                                else:
                                    st.markdown(f"- **Response Time:** Unknown")
                            
                            # Preview cleaned body (body_preview now contains plain text)
                            preview = clean_email_body(email['body_preview'])
                            if len(preview) > 100:
                                preview = preview[:100] + "..."
                            st.markdown(f"- **Preview:** {preview}")
                            st.markdown("")
    
    # Performance insights
    st.markdown("---")
    st.subheader("💡 Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category performance
        category_performance = thread_df.groupby('classification')['completion_days'].agg(['mean', 'count']).reset_index()
        category_performance.columns = ['Category', 'Avg_Days', 'Thread_Count']
        category_performance = category_performance.sort_values('Avg_Days')
        
        st.markdown("**📊 Category Performance (Average Days to Complete):**")
        for _, row in category_performance.iterrows():
            st.markdown(f"- **{row['Category']}**: {row['Avg_Days']:.1f} days ({row['Thread_Count']} threads)")
    
    with col2:
        # Quick vs slow completion analysis
        quick_threads = len(thread_df[thread_df['completion_days'] <= 1])
        medium_threads = len(thread_df[(thread_df['completion_days'] > 1) & (thread_df['completion_days'] <= 7)])
        slow_threads = len(thread_df[thread_df['completion_days'] > 7])
        
        st.markdown("**⚡ Response Speed Analysis:**")
        st.markdown(f"- **Same Day (≤1 day)**: {quick_threads} threads ({quick_threads/len(thread_df)*100:.1f}%)")
        st.markdown(f"- **Within Week (1-7 days)**: {medium_threads} threads ({medium_threads/len(thread_df)*100:.1f}%)")
        st.markdown(f"- **Over Week (>7 days)**: {slow_threads} threads ({slow_threads/len(thread_df)*100:.1f}%)")

def clean_email_body(email_text):
    """
    Clean email body content - now handling plain text instead of HTML
    Since the new body column contains plain text, we focus on cleaning email metadata
    """
    if not email_text or pd.isna(email_text):
        return "No content available"
    
    # Convert to string
    text = str(email_text)
    
    # Since body is now plain text, we don't need BeautifulSoup HTML parsing
    # Just clean email metadata patterns
    patterns_to_remove = [
        # Remove "Sent:" lines with date/time
        r'Sent:\s*\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}.*?\n',
        # Remove "To:" lines
        r'To:\s*.*?<.*?>.*?\n',
        r'To:\s*.*?\n',
        # Remove "From:" lines
        r'From:\s*.*?<.*?>.*?\n',
        r'From:\s*.*?\n',
        # Remove "Cc:" lines
        r'Cc:\s*.*?<.*?>.*?\n',
        r'Cc:\s*.*?\n',
        # Remove "Subject:" lines
        r'Subject:\s*.*?\n',
        # Remove email addresses in angle brackets
        r'<[^@\s]+@[^@\s]+\.[^@\s]+>',
        # Remove standalone email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Remove "Reply-To:" lines
        r'Reply-To:\s*.*?\n',
        # Remove "Date:" lines
        r'Date:\s*.*?\n',
        # Remove "Message-ID:" lines
        r'Message-ID:\s*.*?\n',
        # Remove other common email headers
        r'X-.*?:.*?\n',
        r'MIME-Version:.*?\n',
        r'Content-Type:.*?\n',
        r'Content-Transfer-Encoding:.*?\n',
        # Remove excessive whitespace and newlines
        r'\n\s*\n\s*\n+',
        r'^\s*\n+',
        r'\n+\s*$',
    ]
    
    # Apply all patterns
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '\n', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up remaining artifacts
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip lines that look like email metadata
        if any([
            line.startswith('Sent:'),
            line.startswith('To:'),
            line.startswith('From:'),
            line.startswith('Cc:'),
            line.startswith('bcc:'),
            line.startswith('Subject:'),
            line.startswith('Date:'),
            line.startswith('Reply-To:'),
            re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', line),
            re.match(r'^\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}', line),
            line.startswith('Message-ID:'),
            line.startswith('X-'),
            line.startswith('MIME-Version:'),
            line.startswith('Content-'),
        ]):
            continue
        
        # Skip lines that are just names in angle brackets or email signatures
        if re.match(r'^<.*>$', line) or len(line) < 3:
            continue
        
        cleaned_lines.append(line)
    
    # Join cleaned lines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Final cleanup - remove extra whitespace
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    # If nothing meaningful remains, return a default message
    if not cleaned_text or len(cleaned_text) < 10:
        return "Email content appears to be empty or contains only metadata."
    
    return cleaned_text

def get_status_display(status, status_error=None):
    """Return formatted status display with appropriate emoji and color"""
    if pd.isna(status) or status is None:
        return "❓ Unknown", "secondary"
    
    status = str(status).lower().strip()
    
    status_mapping = {
        'success': ('✅ Success', 'success'),
        'completed': ('✅ Completed', 'success'),
        'processed': ('✅ Processed', 'success'),
        'pending': ('⏳ Pending', 'warning'),
        'processing': ('🔄 Processing', 'info'),
        'in_progress': ('🔄 In Progress', 'info'),
        'failed': ('❌ Failed', 'error'),
        'error': ('❌ Error', 'error'),
        'cancelled': ('🚫 Cancelled', 'secondary'),
        'rejected': ('🚫 Rejected', 'error'),
        'draft': ('📝 Draft', 'secondary'),
        'sent': ('📤 Sent', 'success'),
        'delivered': ('📬 Delivered', 'success'),
        'read': ('👁️ Read', 'info'),
    }
    
    # Get status display or use default
    emoji_status, color = status_mapping.get(status, (f"🔹 {status.title()}", "secondary"))
    
    # Add error info if present
    if status_error and str(status_error).strip() and not pd.isna(status_error):
        emoji_status += f" (Error: {str(status_error)[:50]}{'...' if len(str(status_error)) > 50 else ''})"
    
    return emoji_status, color

def clean_html(html_text):
    """Legacy function for backward compatibility - now uses enhanced cleaning"""
    return clean_email_body(html_text)

# Initialize session state
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_email_id' not in st.session_state:
    st.session_state.selected_email_id = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'dashboard'
# Use different keys for session state storage
if 'stored_category_start_date' not in st.session_state:
    st.session_state.stored_category_start_date = None
if 'stored_category_end_date' not in st.session_state:
    st.session_state.stored_category_end_date = None

# Categories
CATEGORIES = [
    "Initial Enquiry", 
    "Product/Service Information", 
    "Follow-up", 
    "Quotation Submission", 
    "Negotiation", 
    "Approval Process", 
    "Order Confirmation", 
    "Dispatch & Delivery", 
    "Post-Delivery / Feedback"
]

# Main application
def main():
    st.title("📧 Email Classification Dashboard")
    st.markdown("---")
    
    # Initialize database
    engine = init_database_connection()
    if engine is None:
        st.stop()
    
    # Load data (without date filtering)
    df = load_email_data(engine)
    
    if df.empty:
        st.warning("No classified emails found for the selected date range.")
        return
    
    # Navigation buttons in sidebar
    st.sidebar.header("🧭 Navigation")
    
    # Main navigation buttons
    if st.sidebar.button("🏠 Dashboard Overview", key="nav_dashboard"):
        st.session_state.current_view = 'dashboard'
        st.session_state.selected_category = None
        st.session_state.selected_email_id = None
    
    if st.sidebar.button("⏱️ Task Completion Analysis", key="nav_completion"):
        st.session_state.current_view = 'completion'
        st.session_state.selected_category = None
        st.session_state.selected_email_id = None
    
    # Context-sensitive navigation
    if st.session_state.current_view == 'dashboard':
        if st.sidebar.button("🔄 Refresh Data", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()
    
    if st.session_state.selected_category and st.sidebar.button("📂 Back to Category", key="back_to_category"):
        st.session_state.selected_email_id = None
    
    if (st.session_state.selected_category or st.session_state.selected_email_id) and st.sidebar.button("🏠 Back to Main View", key="back_to_main"):
        st.session_state.selected_category = None
        st.session_state.selected_email_id = None
        st.session_state.stored_category_start_date = None
        st.session_state.stored_category_end_date = None
        # Clear status filters as well
        st.session_state.stored_status_filter = "All"
        if hasattr(st.session_state, 'stored_custom_statuses'):
            st.session_state.stored_custom_statuses = []
    
    # Main content based on navigation state
    if st.session_state.selected_email_id:
        show_email_detail(df)
    elif st.session_state.selected_category:
        show_category_emails(df)
    elif st.session_state.current_view == 'completion':
        show_task_completion_analysis(df)
    else:
        show_dashboard_overview(df)

def show_dashboard_overview(df):
    """Show main dashboard with category distribution and status overview"""
    st.header("📊 Email Classification Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Emails",
            value=len(df),
            delta=f"{len(df[df['email_date'] == df['email_date'].max()])} today" if not df.empty else "0 today"
        )
    
    with col2:
        unique_senders = df['from'].nunique() if not df.empty else 0
        st.metric(
            label="Unique Senders",
            value=unique_senders
        )
    
    with col3:
        if not df.empty:
            most_common_category = df['classification'].mode().iloc[0] if not df['classification'].mode().empty else "N/A"
        else:
            most_common_category = "N/A"
        st.metric(
            label="Most Common Category",
            value=most_common_category
        )
    
    with col4:
        # Count emails with attachments (now boolean field)
        emails_with_attachments = len(df[df['attachment'] == 'Yes']) if not df.empty else 0
        st.metric(
            label="Emails with Attachments",
            value=emails_with_attachments
        )
    
    with col5:
        # Email threads metric
        thread_df = calculate_email_thread_completion_time(df)
        st.metric(
            label="Email Threads",
            value=len(thread_df),
            delta=f"Avg: {thread_df['completion_days'].mean():.1f} days" if not thread_df.empty else "No threads"
        )
    
    st.markdown("---")
    
    # Quick access to task completion analysis
    if not calculate_email_thread_completion_time(df).empty:
        st.info("💡 **New Feature**: View task completion time analysis to see how long email conversations take to resolve!")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("⏱️ View Task Completion Analysis", type="primary"):
                st.session_state.current_view = 'completion'
                st.rerun()
    
    # Status Distribution (New Section)
    if not df.empty and 'status' in df.columns:
        st.subheader("📊 Email Status Overview")
        
        # Status distribution
        status_counts = df['status'].fillna('Unknown').value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status pie chart
            fig_status_pie = px.pie(
                status_counts,
                values='Count',
                names='Status',
                title="Email Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_status_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_status_pie, use_container_width=True)
        
        with col2:
            # Status bar chart
            fig_status_bar = px.bar(
                status_counts,
                x='Status',
                y='Count',
                title="Email Status Counts",
                color='Count',
                color_continuous_scale='blues'
            )
            fig_status_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_status_bar, use_container_width=True)
        
        st.markdown("---")
    
    # Category distribution
    if not df.empty:
        category_counts = df['classification'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Category Distribution")
            
            # Pie chart
            fig_pie = px.pie(
                category_counts, 
                values='Count', 
                names='Category',
                title="Email Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Category Counts")
            
            # Bar chart
            fig_bar = px.bar(
                category_counts,
                x='Category',
                y='Count',
                title="Email Counts by Category",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Time series analysis
        st.subheader("📅 Email Trends Over Time")
        
        # Group by date and category
        daily_counts = df.groupby(['email_date', 'classification']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            daily_counts,
            x='email_date',
            y='count',
            color='classification',
            title="Daily Email Volume by Category",
            markers=True
        )
        fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Number of Emails")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Date-wise category table
        st.subheader("📅 Date-wise Email Distribution")
        
        # Create pivot table with dates as rows and categories as columns
        pivot_data = df.groupby(['email_date', 'classification']).size().reset_index(name='count')
        pivot_table = pivot_data.pivot(index='email_date', columns='classification', values='count').fillna(0).astype(int)
        
        # Ensure all categories are represented as columns
        for category in CATEGORIES:
            if category not in pivot_table.columns:
                pivot_table[category] = 0
        
        # Reorder columns to match CATEGORIES order
        pivot_table = pivot_table[CATEGORIES]
        
        # Add a total column
        pivot_table['📊 Total'] = pivot_table.sum(axis=1)
        
        # Sort by date (newest first)
        pivot_table = pivot_table.sort_index(ascending=False)
        
        # Display the table with better formatting
        st.dataframe(
            pivot_table,
            use_container_width=True,
            height=400,
            column_config={
                "📊 Total": st.column_config.NumberColumn(
                    "📊 Total",
                    help="Total emails per day",
                    format="%d",
                ),
                **{category: st.column_config.NumberColumn(
                    category,
                    help=f"Number of {category} emails",
                    format="%d",
                ) for category in CATEGORIES}
            }
        )
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Most Active Day", 
                     pivot_table['📊 Total'].idxmax().strftime('%Y-%m-%d') if not pivot_table.empty else "N/A",
                     f"{pivot_table['📊 Total'].max()} emails" if not pivot_table.empty else "0")
        
        with col2:
            st.metric("📉 Least Active Day", 
                     pivot_table['📊 Total'].idxmin().strftime('%Y-%m-%d') if not pivot_table.empty else "N/A",
                     f"{pivot_table['📊 Total'].min()} emails" if not pivot_table.empty else "0")
        
        with col3:
            avg_daily = pivot_table['📊 Total'].mean() if not pivot_table.empty else 0
            st.metric("📊 Daily Average", 
                     f"{avg_daily:.1f} emails")
        
        st.markdown("---")
        
        # Interactive category selection with date filter
        st.subheader("🎯 Category Details")
        
        # Add filters for categories
        st.markdown("**Filter Options for Category View:**")
        
        # Date range filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**📅 Date Range:**")
        with col2:
            # Use default values from session state if available
            default_start = st.session_state.stored_category_start_date if st.session_state.stored_category_start_date else datetime.now() - timedelta(days=7)
            category_start_date = st.date_input(
                "From Date",
                value=default_start,
                key="category_start_date",
                help="Start date for category email filtering"
            )
        with col3:
            # Use default values from session state if available
            default_end = st.session_state.stored_category_end_date if st.session_state.stored_category_end_date else datetime.now()
            category_end_date = st.date_input(
                "To Date",
                value=default_end,
                key="category_end_date",
                help="End date for category email filtering"
            )
        
        # Status filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**📊 Status Filter:**")
        with col2:
            # Initialize status filter in session state if not exists
            if 'stored_status_filter' not in st.session_state:
                st.session_state.stored_status_filter = "All"
            
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "Completed", "Pending"],
                index=["All", "Completed", "Pending"].index(st.session_state.stored_status_filter) if st.session_state.stored_status_filter in ["All", "Completed", "Pending"] else 0,
                key="status_filter_select",
                help="Filter emails by processing status"
            )
            
            # Update stored value
            st.session_state.stored_status_filter = status_filter
            
        with col3:
            # Custom status selection if "Custom" is selected
            if status_filter == "Custom":
                # Get unique statuses from the dataframe
                unique_statuses = df['status'].fillna('Unknown').unique().tolist()
                
                if 'stored_custom_statuses' not in st.session_state:
                    st.session_state.stored_custom_statuses = []
                
                custom_statuses = st.multiselect(
                    "Select Statuses",
                    options=unique_statuses,
                    default=st.session_state.stored_custom_statuses,
                    key="custom_status_multiselect",
                    help="Select specific statuses to include"
                )
                
                st.session_state.stored_custom_statuses = custom_statuses
            else:
                custom_statuses = []
        
        # Update stored values only when they change
        if category_start_date != st.session_state.stored_category_start_date:
            st.session_state.stored_category_start_date = category_start_date
        if category_end_date != st.session_state.stored_category_end_date:
            st.session_state.stored_category_end_date = category_end_date
        
        # Apply date filter first
        filtered_df = df[
            (df['email_date'] >= pd.to_datetime(category_start_date)) & 
            (df['email_date'] <= pd.to_datetime(category_end_date))
        ]
        
        # Apply status filter
        if status_filter != "All":
            if status_filter == "Completed":
                # Filter for completed/successful statuses
                filtered_df = filtered_df[
                    filtered_df['status'].str.lower().str.contains('success|completed|processed|sent|delivered', na=False)
                ]
            elif status_filter == "Pending":
                # Filter for pending/processing statuses
                filtered_df = filtered_df[
                    filtered_df['status'].str.lower().str.contains('pending|processing|in_progress', na=False)
                ]
            elif status_filter == "Failed/Error":
                # Filter for failed/error statuses
                filtered_df = filtered_df[
                    filtered_df['status'].str.lower().str.contains('failed|error|cancelled|rejected', na=False)
                ]
            elif status_filter == "Custom" and custom_statuses:
                # Filter for custom selected statuses
                filtered_df = filtered_df[
                    filtered_df['status'].fillna('Unknown').isin(custom_statuses)
                ]
        
        if not filtered_df.empty:
            filtered_category_counts = filtered_df['classification'].value_counts().reset_index()
            filtered_category_counts.columns = ['Category', 'Count']
            
            # Show info about the filters applied
            filter_info = f"📅 Date: {category_start_date} to {category_end_date}"
            if status_filter != "All":
                if status_filter == "Custom" and custom_statuses:
                    filter_info += f" | 📊 Status: {', '.join(custom_statuses)}"
                else:
                    filter_info += f" | 📊 Status: {status_filter}"
            
            st.info(f"Showing category counts with filters - {filter_info} ({len(filtered_df)} total emails)")
            
            # Add summary of status distribution in filtered data
            if 'status' in filtered_df.columns and status_filter == "All":
                status_summary = filtered_df['status'].fillna('Unknown').value_counts()
                if len(status_summary) > 0:
                    status_text = " | ".join([f"{status}: {count}" for status, count in status_summary.head(5).items()])
                    if len(status_summary) > 5:
                        status_text += f" | +{len(status_summary) - 5} more"
                    st.markdown(f"**Status Distribution:** {status_text}")
            
            st.markdown("**Click on a category below to view filtered emails:**")
            
            # Create clickable buttons for each category with filtered counts
            cols_per_row = 3
            filtered_categories_dict = dict(zip(filtered_category_counts['Category'], filtered_category_counts['Count']))
            
            # Ensure all categories are shown, even with 0 counts
            all_categories_with_counts = []
            for category in CATEGORIES:
                count = filtered_categories_dict.get(category, 0)
                all_categories_with_counts.append({'Category': category, 'Count': count})
            
            for i in range(0, len(all_categories_with_counts), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(all_categories_with_counts):
                        cat_data = all_categories_with_counts[i + j]
                        category = cat_data['Category']
                        count = cat_data['Count']
                        
                        with col:
                            # Disable button if no emails in this category for the selected filters
                            button_disabled = count == 0
                            button_text = f"📁 {category}\n({count} emails)"
                            
                            if count == 0:
                                st.button(
                                    button_text,
                                    key=f"cat_disabled_{category}",
                                    use_container_width=True,
                                    disabled=True,
                                    help=f"No emails found in '{category}' for the selected filters"
                                )
                            else:
                                if st.button(
                                    button_text,
                                    key=f"cat_{category}",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_category = category
                                    st.rerun()
        else:
            st.warning("No emails found matching the selected filters.")

def show_category_emails(df):
    """Show emails for selected category with date filtering and task completion info"""
    category = st.session_state.selected_category
    
    # Apply date filter if set
    if st.session_state.stored_category_start_date and st.session_state.stored_category_end_date:
        start_date = pd.to_datetime(st.session_state.stored_category_start_date)
        end_date = pd.to_datetime(st.session_state.stored_category_end_date)
        category_emails = df[
            (df['classification'] == category) & 
            (df['email_date'] >= start_date) & 
            (df['email_date'] <= end_date)
        ].copy()
        
        date_filter_text = f" (from {st.session_state.stored_category_start_date} to {st.session_state.stored_category_end_date})"
    else:
        category_emails = df[df['classification'] == category].copy()
        date_filter_text = ""
    
    # Apply status filter if set
    if hasattr(st.session_state, 'stored_status_filter') and st.session_state.stored_status_filter != "All":
        status_filter = st.session_state.stored_status_filter
        
        if status_filter == "Completed":
            category_emails = category_emails[
                category_emails['status'].str.lower().str.contains('success|completed|processed|sent|delivered', na=False)
            ]
            status_filter_text = " | Status: Completed"
        elif status_filter == "Pending":
            category_emails = category_emails[
                category_emails['status'].str.lower().str.contains('pending|processing|in_progress', na=False)
            ]
            status_filter_text = " | Status: Pending"
        elif status_filter == "Failed/Error":
            category_emails = category_emails[
                category_emails['status'].str.lower().str.contains('failed|error|cancelled|rejected', na=False)
            ]
            status_filter_text = " | Status: Failed/Error"
        elif status_filter == "Custom" and hasattr(st.session_state, 'stored_custom_statuses') and st.session_state.stored_custom_statuses:
            category_emails = category_emails[
                category_emails['status'].fillna('Unknown').isin(st.session_state.stored_custom_statuses)
            ]
            status_filter_text = f" | Status: {', '.join(st.session_state.stored_custom_statuses)}"
        else:
            status_filter_text = ""
    else:
        status_filter_text = ""
    
    st.header(f"📁 {category}")
    st.markdown(f"**{len(category_emails)} emails found{date_filter_text}{status_filter_text}**")
    
    # Calculate thread completion times for this category
    category_threads = calculate_email_thread_completion_time(category_emails)
    
    # Show task completion summary for this category
    if not category_threads.empty:
        st.markdown("### ⏱️ Task Completion Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Email Threads", len(category_threads))
        with col2:
            avg_completion = category_threads['completion_days'].mean()
            st.metric("Avg Completion", f"{avg_completion:.1f} days")
        with col3:
            fastest = category_threads['completion_days'].min()
            st.metric("Fastest", f"{fastest:.1f} days")
        with col4:
            longest = category_threads['completion_days'].max()
            st.metric("Longest", f"{longest:.1f} days")
    
    # Show current filters applied
    if st.session_state.stored_category_start_date and st.session_state.stored_category_end_date:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            filter_info = f"📅 Date: {st.session_state.stored_category_start_date} to {st.session_state.stored_category_end_date}"
            if hasattr(st.session_state, 'stored_status_filter') and st.session_state.stored_status_filter != "All":
                if st.session_state.stored_status_filter == "Custom" and hasattr(st.session_state, 'stored_custom_statuses'):
                    filter_info += f" | 📊 Status: {', '.join(st.session_state.stored_custom_statuses) if st.session_state.stored_custom_statuses else 'None selected'}"
                else:
                    filter_info += f" | 📊 Status: {st.session_state.stored_status_filter}"
            st.info(f"Applied filters: {filter_info}")
        with col2:
            if st.button("📅 Change Filters", key="change_filters"):
                st.session_state.selected_category = None  # Go back to main view to change filters
                st.rerun()
        with col3:
            if st.button("🗑️ Clear All Filters", key="clear_all_filters"):
                st.session_state.stored_category_start_date = None
                st.session_state.stored_category_end_date = None
                st.session_state.stored_status_filter = "All"
                if hasattr(st.session_state, 'stored_custom_statuses'):
                    st.session_state.stored_custom_statuses = []
                st.rerun()
    
    if category_emails.empty:
        st.warning(f"No emails found in category '{category}' for the selected criteria.")
        return
    
    # Add normalized subject for thread identification
    category_emails['normalized_subject'] = category_emails['subject'].apply(normalize_subject)
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Email List")
    with col2:
        sort_option = st.selectbox(
            "Sort by:",
            ["Newest First", "Oldest First", "Subject A-Z", "Subject Z-A"],
            key="sort_emails"
        )
    
    # Apply sorting
    if sort_option == "Newest First":
        category_emails = category_emails.sort_values('received_time', ascending=False)
    elif sort_option == "Oldest First":
        category_emails = category_emails.sort_values('received_time', ascending=True)
    elif sort_option == "Subject A-Z":
        category_emails = category_emails.sort_values('subject', ascending=True)
    else:  # Subject Z-A
        category_emails = category_emails.sort_values('subject', ascending=False)
    
    # Create thread lookup for completion times
    thread_lookup = {}
    if not category_threads.empty:
        for _, thread in category_threads.iterrows():
            for email_id in thread['emails_in_thread']:
                thread_lookup[email_id] = thread
    
    # Display emails in cards with thread completion info
    for idx, email in category_emails.iterrows():
        with st.container():
            st.markdown("---")
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Email preview
                subject = email['subject'] if email['subject'] else "No Subject"
                sender = email['from'] if email['from'] else "Unknown Sender"
                received_time = email['received_time'].strftime("%Y-%m-%d %H:%M")
                
                # Check if this email is part of a thread
                is_reply = bool(re.search(r'^(RE|FW|FWD):', str(subject), re.IGNORECASE))
                thread_indicator = " 🔄" if is_reply else ""
                
                st.markdown(f"**📧 {subject}{thread_indicator}**")
                st.markdown(f"**From:** {sender}")
                st.markdown(f"**Received:** {received_time}")
                st.markdown(f"**Category:** {email['classification']}")
                
                # Show preview of body (now plain text from body column)
                body_preview = clean_email_body(email['body_preview'])
                if len(body_preview) > 150:
                    body_preview = body_preview[:150] + "..."
                st.markdown(f"📝 Preview: {body_preview}")
                
                # Show attachment info (now boolean field)
                if email['attachment'] == 'Yes':
                    st.markdown("📎 Has attachments")
            
            with col2:
                # Show status with color coding
                if 'status' in email and email['status'] is not None:
                    status_display, status_color = get_status_display(email['status'], email.get('status_error'))
                    
                    # Use different colored containers based on status
                    if status_color == 'success':
                        st.success(f"Status: {status_display}")
                    elif status_color == 'error':
                        st.error(f"Status: {status_display}")
                    elif status_color == 'warning':
                        st.warning(f"Status: {status_display}")
                    elif status_color == 'info':
                        st.info(f"Status: {status_display}")
                    else:
                        st.markdown(f"Status: {status_display}")
                else:
                    st.markdown("Status: ❓ Unknown")
                
                # Show thread completion time if email is part of a thread
                if email['id'] in thread_lookup:
                    thread_info = thread_lookup[email['id']]
                    days = thread_info['completion_days']
                    
                    st.markdown("**⏱️ Thread Completion:**")
                    if days <= 1:
                        st.success(f"⚡ {days} days")
                        st.caption(f"({thread_info['completion_hours']:.1f} hours)")
                    elif days <= 7:
                        st.info(f"📅 {days} days")
                    elif days <= 30:
                        st.warning(f"📆 {days} days")
                    else:
                        st.error(f"⏳ {days} days")
                    
                    st.caption(f"Thread: {thread_info['email_count']} emails")
                    
                    # Show thread details
                    try:
                        if pd.notna(thread_info['first_email_date']):
                            start_str = thread_info['first_email_date'].strftime('%m-%d %H:%M')
                        else:
                            start_str = "Unknown"
                        
                        if pd.notna(thread_info['last_email_date']):
                            latest_str = thread_info['last_email_date'].strftime('%m-%d %H:%M')
                        else:
                            latest_str = "Unknown"
                        
                        st.markdown(f"**Started:** {start_str}")
                        st.markdown(f"**Latest:** {latest_str}")
                    except (AttributeError, TypeError, ValueError):
                        st.markdown(f"**Started:** Invalid date")
                        st.markdown(f"**Latest:** Invalid date")
                else:
                    st.markdown("**⏱️ Single Email**")
                    st.caption("No thread data")
            
            with col3:
                if st.button(
                    "📖 View Details",
                    key=f"view_{email['id']}",
                    use_container_width=True
                ):
                    st.session_state.selected_email_id = email['id']
                    st.rerun()

def show_email_detail(df):
    """Show detailed view of selected email with thread analysis"""
    email_id = st.session_state.selected_email_id
    email = df[df['id'] == email_id].iloc[0] if not df[df['id'] == email_id].empty else None
    
    if email is None:
        st.error("Email not found")
        return
    
    st.header("📧 Email Details")
    
    # Check if this email is part of a thread
    normalized_subject = normalize_subject(email['subject'])
    thread_emails = df[df['subject'].apply(normalize_subject) == normalized_subject].sort_values('received_time')
    
    # Thread analysis if multiple emails exist
    if len(thread_emails) > 1:
        st.subheader("🔄 Email Thread Analysis")
        
        # Calculate completion time
        first_email = thread_emails.iloc[0]
        last_email = thread_emails.iloc[-1]
        time_diff = last_email['received_time'] - first_email['received_time']
        completion_days = time_diff.total_seconds() / (24 * 3600)
        completion_hours = completion_days * 24
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Thread Emails", len(thread_emails))
        with col2:
            reply_count = len(thread_emails[thread_emails['subject'].str.contains(r'^(RE|FW|FWD):', case=False, na=False)])
            st.metric("Replies", reply_count)
        with col3:
            if completion_days <= 1:
                st.success(f"⚡ Completion Time\n{completion_days:.1f} days")
            elif completion_days <= 7:
                st.info(f"📅 Completion Time\n{completion_days:.1f} days")
            else:
                st.warning(f"📆 Completion Time\n{completion_days:.1f} days")
        with col4:
            st.metric("Total Hours", f"{completion_hours:.1f}")
        
        # Thread timeline
        with st.expander("📅 Thread Timeline", expanded=True):
            for i, (_, thread_email) in enumerate(thread_emails.iterrows()):
                is_current = thread_email['id'] == email['id']
                
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    marker = "🔵" if is_current else "⚪"
                    is_reply = bool(re.search(r'^(RE|FW|FWD):', str(thread_email['subject']), re.IGNORECASE))
                    reply_marker = " 🔄" if is_reply else " 📧"
                    
                    st.markdown(f"{marker} **Email {i+1}**{reply_marker}")
                    st.markdown(f"From: {thread_email['from'] if thread_email['from'] else 'Unknown'}")
                    st.markdown(f"Subject: {thread_email['subject'] if thread_email['subject'] else 'No Subject'}")
                
                with col_b:
                    # Safe datetime formatting
                    try:
                        if pd.notna(thread_email['received_time']):
                            st.markdown(f"**{thread_email['received_time'].strftime('%Y-%m-%d')}**")
                            st.markdown(f"{thread_email['received_time'].strftime('%H:%M')}")
                        else:
                            st.markdown("**Unknown Date**")
                            st.markdown("--:--")
                    except (AttributeError, TypeError, ValueError):
                        st.markdown("**Invalid Date**")
                        st.markdown("--:--")
                
                with col_c: 
                    if i > 0:
                        prev_email = thread_emails.iloc[i-1]
                        
                        # Safe time difference calculation
                        try:
                            if pd.notna(thread_email['received_time']) and pd.notna(prev_email['received_time']):
                                response_time = thread_email['received_time'] - prev_email['received_time']
                                response_hours = response_time.total_seconds() / 3600
                                
                                if response_hours < 1:
                                    st.success(f"⚡ {response_hours*60:.0f} min")
                                elif response_hours < 24:
                                    st.info(f"🕐 {response_hours:.1f} hrs")
                                else:
                                    st.warning(f"📅 {response_hours/24:.1f} days")
                            else:
                                st.markdown("**Unknown**")
                        except (AttributeError, TypeError, ValueError):
                            st.markdown("**Error**")
                    else:
                        st.markdown("**Thread Start**")
                
                if i < len(thread_emails) - 1:
                    st.markdown("↓")
        
        st.markdown("---")
    
    # Email metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Email Information")
        st.markdown(f"**Subject:** {email['subject'] if email['subject'] else 'No Subject'}")
        st.markdown(f"**From:** {email['from'] if email['from'] else 'Unknown'}")
        st.markdown(f"**Received:** {email['received_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"**Category:** {email['classification']}")
        
        # Thread information
        if len(thread_emails) > 1:
            st.markdown(f"**Thread Position:** Email {list(thread_emails['id']).index(email['id']) + 1} of {len(thread_emails)}")
            st.markdown(f"**Normalized Subject:** {normalized_subject}")
        
        # Status Information Section
        st.markdown("### 📊 Status Information")
        if 'status' in email and email['status'] is not None:
            status_display, status_color = get_status_display(email['status'], email.get('status_error'))
            
            # Display status with appropriate styling
            if status_color == 'success':
                st.success(f"**Current Status:** {status_display}")
            elif status_color == 'error':
                st.error(f"**Current Status:** {status_display}")
            elif status_color == 'warning':
                st.warning(f"**Current Status:** {status_display}")
            elif status_color == 'info':
                st.info(f"**Current Status:** {status_display}")
            else:
                st.markdown(f"**Current Status:** {status_display}")
            
            # Show status update time if available
            if 'status_updated_at' in email and email['status_updated_at'] is not None and not pd.isna(email['status_updated_at']):
                st.markdown(f"**Status Updated:** {email['status_updated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show status error details if present
            if 'status_error' in email and email['status_error'] is not None and str(email['status_error']).strip() and not pd.isna(email['status_error']):
                with st.expander("⚠️ Status Error Details", expanded=False):
                    st.error(f"Error Details: {email['status_error']}")
        else:
            st.markdown("**Current Status:** ❓ Unknown")
        
    with col2:
        st.markdown("### 👥 Recipients")
        if email['cc'] and str(email['cc']).strip():
            st.markdown(f"**cc:** {email['cc']}")
        else:
            st.markdown("**cc:** None")
            
        if email['bcc'] and str(email['bcc']).strip():
            st.markdown(f"**bcc:** {email['bcc']}")
        else:
            st.markdown("**bcc:** None")
            
        if email['attachment'] == 'Yes':
            st.markdown(f"**attachments:** Yes")
        else:
            st.markdown("**attachments:** None")
    
    st.markdown("---")
    
    # Email body - now displaying plain text content
    st.markdown("### 📝 Email Content")
    
    body_content = clean_email_body(email['body_preview'])
    
    # Show full content in a text area for safe display
    st.text_area(
        "Email Content:",
        value=body_content,
        height=300,
        disabled=True,
        label_visibility="collapsed"
    )
    
    # Show original content for comparison (optional - can be removed in production)
    with st.expander("🔍 View Raw Content (for debugging)", expanded=False):
        raw_content = str(email['body_preview']) if email['body_preview'] else "No content"
        st.text_area(
            "Raw Content:",
            value=raw_content,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
    
    # Status Management Section
    if 'status' in email:
        with st.expander("🔧 Status Management", expanded=False):
            st.markdown("### Update Email Status")
            
            # Status update form
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_status = st.selectbox(
                    "Select New Status:",
                    ["success", "pending", "processing", "failed", "error", "cancelled", "sent", "delivered", "read", "draft"],
                    index=0 if email['status'] is None else (
                        ["success", "pending", "processing", "failed", "error", "cancelled", "sent", "delivered", "read", "draft"].index(str(email['status']).lower()) 
                        if str(email['status']).lower() in ["success", "pending", "processing", "failed", "error", "cancelled", "sent", "delivered", "read", "draft"] 
                        else 0
                    ),
                    key="new_status_select"
                )
            
            with col2:
                if st.button("🔄 Update Status", type="primary", use_container_width=True):
                    # Here you would update the database
                    st.success(f"Status would be updated to: {new_status}")
                    # Database update query would go here:
                    # UPDATE public.emailsdatalogger SET status = new_status, status_updated_at = NOW() WHERE id = email['id']
                    st.info("💡 Database update functionality would be implemented here")
            
            # Error message input (if status is error or failed)
            if new_status in ['error', 'failed']:
                error_message = st.text_area(
                    "Error Message (optional):",
                    value="",
                    height=100,
                    help="Provide details about the error",
                    key="status_error_input"
                )
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📧 Reply", use_container_width=True):
            st.info("Reply functionality would be implemented here")
    
    with col2:
        if st.button("📤 Forward", use_container_width=True):
            st.info("Forward functionality would be implemented here")
    
    with col3:
        # Category reassignment
        new_category = st.selectbox(
            "Reassign Category:",
            CATEGORIES,
            index=CATEGORIES.index(email['classification']) if email['classification'] in CATEGORIES else 0,
            key="reassign_category"
        )
        
        if st.button("💾 Update Category"):
            # Here you would update the database
            st.success(f"Category updated to: {new_category}")
            # You can add database update logic here
            # UPDATE public.emailsdatalogger SET classification = new_category WHERE id = email['id']

def show_status_overview(df):
    """Show a dedicated status overview page"""
    st.header("📊 Email Status Overview")
    
    if df.empty or 'status' not in df.columns:
        st.warning("No status data available.")
        return
    
    # Status summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_emails = len(df)
    success_emails = len(df[df['status'].str.lower().str.contains('success|completed|processed|sent|delivered', na=False)])
    pending_emails = len(df[df['status'].str.lower().str.contains('pending|processing|in_progress', na=False)])
    failed_emails = len(df[df['status'].str.lower().str.contains('failed|error', na=False)])
    
    with col1:
        st.metric("Total Emails", total_emails)
    with col2:
        success_rate = (success_emails / total_emails * 100) if total_emails > 0 else 0
        st.metric("Successful", success_emails, f"{success_rate:.1f}%")
    with col3:
        pending_rate = (pending_emails / total_emails * 100) if total_emails > 0 else 0
        st.metric("Pending/Processing", pending_emails, f"{pending_rate:.1f}%")
    with col4:
        fail_rate = (failed_emails / total_emails * 100) if total_emails > 0 else 0
        st.metric("Failed/Error", failed_emails, f"{fail_rate:.1f}%")
    
    st.markdown("---")
    
    # Detailed status table
    st.subheader("📋 Detailed Status Breakdown")
    
    # Create status summary dataframe
    status_summary = df.groupby('status').agg({
        'id': 'count',
        'status_updated_at': 'max',
        'status_error': lambda x: x.notna().sum()
    }).reset_index()
    
    status_summary.columns = ['Status', 'Count', 'Last Updated', 'Error Count']
    
    # Sort by count descending
    status_summary = status_summary.sort_values('Count', ascending=False)
    
    # Display with formatting
    st.dataframe(
        status_summary,
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", help="Email processing status"),
            "Count": st.column_config.NumberColumn("Count", help="Number of emails with this status"),
            "Last Updated": st.column_config.DatetimeColumn("Last Updated", help="Most recent status update"),
            "Error Count": st.column_config.NumberColumn("Errors", help="Number of emails with errors in this status"),
        }
    )
    
    # Emails with errors section
    if failed_emails > 0:
        st.subheader("⚠️ Emails with Errors")
        
        error_emails = df[df['status'].str.lower().str.contains('failed|error', na=False)].copy()
        
        if not error_emails.empty:
            error_display = error_emails[['subject', 'from', 'status', 'status_error', 'status_updated_at']].copy()
            error_display = error_display.sort_values('status_updated_at', ascending=False, na_position='last')
            
            st.dataframe(
                error_display,
                use_container_width=True,
                column_config={
                    "subject": st.column_config.TextColumn("subject", width="medium"),
                    "from": st.column_config.TextColumn("from", width="medium"), 
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "status_error": st.column_config.TextColumn("Error Details", width="large"),
                    "status_updated_at": st.column_config.DatetimeColumn("Error Time", width="medium"),
                }
            )

if __name__ == "__main__":
    main()