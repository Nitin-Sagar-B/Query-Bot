import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
import time
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_extraction_chain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks import StreamlitCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
SERPAPI_KEY = "8247ff32a057d8b230e073068f3da30e51261fe7535e1bc0468064bedffd8a3c"
GEMINI_KEY = "AIzaSyDCsW36FPu38wqYzPRHSxaeHpgvC5NcAWw"

# Initialize Langchain components
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_KEY)
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)

# Configure page
st.set_page_config(
    page_title="AI Information Extraction Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tomorrow:wght@400;500;600&display=swap');
    
    /* Main theme colors */
    :root {
        --background-color: #1E1E1E;
        --text-color: #E0E0E0;
        --accent-color: #BB86FC;
        --secondary-color: #03DAC6;
        --error-color: #CF6679;
    }
    
    /* Override Streamlit's default styles */
    .stApp {
        background-color: var(--background-color);
    }
    
    .stMarkdown, .stText, p, span {
        color: var(--text-color) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-family: 'Tomorrow', sans-serif;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--accent-color);
        color: black;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-family: 'Tomorrow', sans-serif;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color);
        color: black;
    }
    
    /* Additional styles remain the same as in original CSS */
    </style>
""", unsafe_allow_html=True)

class ExtractionSchema(BaseModel):
    """Dynamic schema for information extraction"""
    fields: Dict[str, str] = Field(description="Extracted field values")
    confidence: float = Field(description="Confidence score of extraction")
    sources: List[str] = Field(description="Source URLs used for extraction")

class InformationExtractor:
    def __init__(self, llm, search, batch_size: int = 5, max_retries: int = 3):
        self.llm = llm
        self.search = search
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def perform_search(self, query: str) -> List[Dict]:
        """Enhanced search with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                results = self.search.run(query)
                return [{"snippet": results, "link": "search_result"}]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Search failed after {self.max_retries} attempts: {str(e)}")
                    return []
                time.sleep(2 ** attempt)
        return []

    def extract_information(self, entity: str, prompt: str, fields: List[str]) -> Dict:
        """Extract information using Langchain components"""
        try:
            # Generate search query
            search_query = f"Find information about {entity}: {prompt.replace('{entity}', entity)}"
            search_results = self.perform_search(search_query)
            
            if not search_results:
                return {
                    "entity": entity,
                    "status": "Failed",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **{field: "Not found" for field in fields}
                }

            # Prepare context from search results
            context = "\n".join([result.get('snippet', '') for result in search_results])

            # Create extraction prompt
            prompt_template = PromptTemplate(
                template="""
                Based on this context about {entity}, please extract the following information:
                {fields}
                
                Context:
                {context}
                
                Provide only the requested information in a clean JSON format with just the field values.
                If a piece of information is not found, use "Not found".
                """,
                input_variables=["entity", "fields", "context"]
            )

            # Create and run extraction chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(
                entity=entity,
                fields="\n".join([f"- {field}" for field in fields]),
                context=context
            )

            # Process and clean result
            cleaned_result = self._clean_response(result, fields)
            cleaned_result["entity"] = entity
            cleaned_result["status"] = "Success"
            cleaned_result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return cleaned_result

        except Exception as e:
            logger.error(f"Extraction error for {entity}: {str(e)}")
            return {
                "entity": entity,
                "status": "Failed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{field: "Not found" for field in fields}
            }

    def _clean_response(self, response: str, fields: List[str]) -> Dict:
        """Clean and standardize extraction response"""
        try:
            # Remove markdown formatting if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1]

            # Parse JSON response
            cleaned = json.loads(response.strip())
            
            # Ensure all requested fields are present
            result = {}
            for field in fields:
                result[field] = cleaned.get(field, "Not found")
            
            return result

        except json.JSONDecodeError:
            logger.error(f"JSON parsing error: {response}")
            return {field: "Not found" for field in fields}
        except Exception as e:
            logger.error(f"Error in response cleaning: {str(e)}")
            return {field: "Not found" for field in fields}

def parse_fields(prompt: str) -> List[str]:
    """Parse fields from prompt"""
    fields = []
    if "and" in prompt.lower():
        field_list = [f.strip() for f in prompt.lower().split("and")]
        for field in field_list:
            field = field.replace("get", "").replace("the", "").replace("for", "").replace("{entity}", "").strip()
            if field:
                fields.append(field.replace(" ", "_").replace("-", "_"))
    return fields

def process_data_batch(df: pd.DataFrame, extractor: InformationExtractor, 
                      entity_column: str, prompt: str, fields: List[str]) -> pd.DataFrame:
    """Process data in batches with progress tracking"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_entities = len(df)
    for idx in range(0, total_entities, extractor.batch_size):
        batch = df.iloc[idx:min(idx + extractor.batch_size, total_entities)]
        
        for _, row in batch.iterrows():
            entity = row[entity_column]
            status_text.text(f"Processing: {entity} ({idx + 1}/{total_entities})")
            
            result = extractor.extract_information(entity, prompt, fields)
            results.append(result)
            
            progress = (idx + 1) / total_entities
            progress_bar.progress(progress)
            
        time.sleep(1)  # Rate limiting
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)

def parse_fields(prompt: str) -> Dict[str, str]:
    """Parse fields from prompt and create schema"""
    fields = {}
    if "and" in prompt.lower():
        field_list = [f.strip() for f in prompt.lower().split("and")]
        for field in field_list:
            field = field.replace("get", "").replace("the", "").replace("for", "").replace("{entity}", "").strip()
            if field:
                field_key = field.replace(" ", "_").replace("-", "_")
                fields[field_key] = f"Extracted {field}"
    return fields

def main():
    # Initialize session state
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        with st.expander("🔧 Processing Options"):
            batch_size = st.slider("Batch Size", min_value=1, max_value=10, value=5)
            retry_count = st.number_input("Max Retries", min_value=1, max_value=5, value=3)
            
        with st.expander("🤖 Model Configuration"):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Main interface
    st.title("🤖 QueryBot: AI Information Extraction Agent")
    
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        1. Upload a CSV file with your data
        2. Select the column containing entities to search
        3. Enter a search prompt using {entity} as placeholder
        4. Configure processing options in the sidebar
        5. Click 'Process Data' to start extraction
        
        Tips:
        - Use "and" to extract multiple fields
        - Example: "Get the email and location for {entity}"
        - Higher confidence threshold = more reliable results
        """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded file is empty.")
                return
                
            col1, col2 = st.columns(2)
            
            with col1:
                entity_column = st.selectbox("Select entity column:", df.columns.tolist())
                
            with col2:
                prompt_template = st.text_input(
                    "Enter search prompt:",
                    "Get the email and headquarters location for {entity}"
                )

            # Data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head())
                col1, col2 = st.columns(2)
                col1.metric("Total Entities", len(df))
                col2.metric("Unique Entities", df[entity_column].nunique())

            if st.button("🚀 Process Data"):
                # Parse fields
                fields = parse_fields(prompt_template)
                if not fields:
                    st.error("Could not identify fields to extract from prompt.")
                    return

                # Initialize extractor
                extractor = InformationExtractor(
                    llm=llm,
                    search=search,
                    batch_size=batch_size,
                    max_retries=retry_count
                )

                with st.spinner("Processing data..."):
                    # Process data
                    results_df = process_data_batch(
                        df=df,
                        extractor=extractor,
                        entity_column=entity_column,
                        prompt=prompt_template,
                        fields=fields
                    )

                    # Reorder columns to put entity and status first
                    cols = ['entity', 'status', 'timestamp'] + [col for col in results_df.columns 
                        if col not in ['entity', 'status', 'timestamp']]
                    results_df = results_df[cols]

                    # Display results
                    st.success("Processing complete! 🎉")
                    
                    with st.expander("📊 Results Analysis", expanded=True):
                        # Calculate metrics
                        success_rate = (results_df['status'] == 'Success').mean() * 100
                        processed_count = len(results_df)
                        failed_count = len(results_df[results_df['status'] == 'Failed'])

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Processed", processed_count)
                        col2.metric("Success Rate", f"{success_rate:.1f}%")
                        col3.metric("Failed", failed_count)

                        # Show results table
                        st.dataframe(results_df)

                    # Export options
                    st.subheader("📥 Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"extracted_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"extracted_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # Update processing history
                    st.session_state.processing_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "entities_processed": len(df),
                        "success_rate": success_rate,
                        "failed_count": failed_count})

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()