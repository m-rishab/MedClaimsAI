import os
import sys
import numpy as np
import pandas as pd
import faiss
import PyPDF2
import json
import pickle
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import warnings

# Import the specialized healthcare prompt
from prompt import HEALTHCARE_CLAIMS_SPECIALIST_PROMPT

warnings.filterwarnings('ignore')

# Terminal colors for better UX
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[35m'
    YELLOW = '\033[33m'

def print_colored(text, color):
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_code(text):
    print(f"{Colors.PURPLE}🔍 {text}{Colors.ENDC}")

# Enhanced Configuration
class EnhancedRAGConfig:
    def __init__(self):
        # Google Gemini API Configuration
        self.gemini_api_key = "AIzaSyDJD3siTZGjPXsZIgHJkKyc5VEQPhRXo-c"
        self.gemini_model = "gemini-1.5-flash"
        
        # Model Parameters - Enhanced for better responses
        self.temperature = 0.3
        self.max_tokens = 750  # Increased for comprehensive responses
        self.top_p = 0.85
        self.top_k = 25
        
        # RAG Parameters - Optimized for healthcare content
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.chunk_size = 400  # Increased for better context
        self.chunk_overlap = 60  # More overlap for healthcare codes
        self.top_k_results = 5  # More sources for complex queries
        
        # FAISS Parameters
        self.similarity_threshold = 0.25  # Lower threshold for edge cases
        self.normalize_embeddings = True
        
        # New Enhancement Parameters
        self.enable_code_extraction = True
        self.enable_query_enhancement = True
        self.enable_smart_chunking = True
        self.confidence_threshold_warning = 0.4
        self.include_suggestions = True
        self.highlight_codes = True

config = EnhancedRAGConfig()

# Code Extraction and Query Enhancement
class HealthcareCodeExtractor:
    """
    Advanced healthcare code extraction and query enhancement
    """
    
    def __init__(self):
        self.code_patterns = {
            'carc': r'(?:CARC|carc)\s*[-\s]*(\d+)',
            'rarc': r'(?:RARC|rarc)\s*[-\s]*([A-Za-z]\d+)',
            'co_code': r'(?:CO|co)[-\s]*(\d+)',
            'pr_code': r'(?:PR|pr)[-\s]*(\d+)',
            'oa_code': r'(?:OA|oa)[-\s]*(\d+)',
            'pi_code': r'(?:PI|pi)[-\s]*(\d+)',
            'cr_code': r'(?:CR|cr)[-\s]*(\d+)',
            'mixed_codes': r'(\d+),\s*(\d+),\s*([A-Za-z]\d+)',
            'standalone_numeric': r'\b(\d{1,3})\b'
        }
        
        # Common healthcare terms for context enhancement
        self.healthcare_terms = [
            'denial', 'appeal', 'resubmit', 'documentation', 'medical necessity',
            'prior authorization', 'EOB', 'remittance', 'adjustment', 'coordination',
            'deductible', 'copay', 'coinsurance', 'benefits', 'coverage',
            'timely filing', 'duplicate', 'provider', 'patient responsibility'
        ]
    
    def extract_healthcare_codes(self, query: str) -> Dict[str, List[str]]:
        """Extract specific healthcare codes from user queries"""
        extracted_codes = {
            'carc_codes': [],
            'rarc_codes': [],
            'co_codes': [],
            'pr_codes': [],
            'oa_codes': [],
            'pi_codes': [],
            'cr_codes': [],
            'mixed_codes': [],
            'standalone_codes': []
        }
        
        for code_type, pattern in self.code_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if code_type == 'mixed_codes':
                extracted_codes[code_type] = matches
            else:
                extracted_codes[code_type.replace('_code', '_codes')] = matches
        
        return extracted_codes
    
    def enhance_query_with_codes(self, original_query: str, extracted_codes: Dict) -> str:
        """Enhance the query by adding context about extracted codes"""
        enhancements = []
        
        if extracted_codes['carc_codes']:
            codes_str = ', '.join(extracted_codes['carc_codes'])
            enhancements.append(f"Specifically explain CARC codes: {codes_str}")
        
        if extracted_codes['rarc_codes']:
            codes_str = ', '.join(extracted_codes['rarc_codes'])
            enhancements.append(f"Include detailed RARC code explanations: {codes_str}")
        
        if extracted_codes['mixed_codes']:
            for mixed in extracted_codes['mixed_codes']:
                enhancements.append(f"Analyze this code combination: {', '.join(mixed)}")
        
        # Detect intent from query
        query_lower = original_query.lower()
        if 'appeal' in query_lower:
            enhancements.append("Include appeal process and requirements")
        if 'resubmit' in query_lower:
            enhancements.append("Provide resubmission guidelines")
        if 'difference' in query_lower or 'vs' in query_lower:
            enhancements.append("Provide clear comparison and distinctions")
        
        if enhancements:
            enhanced_query = original_query + "\n\nSpecific requirements:\n" + "\n".join(enhancements)
            return enhanced_query
        
        return original_query
    
    def get_query_suggestions(self, current_query: str, chat_history: List[Dict]) -> List[str]:
        """Generate contextual query suggestions"""
        suggestions = []
        extracted_codes = self.extract_healthcare_codes(current_query)
        
        # Code-specific suggestions
        if extracted_codes['carc_codes']:
            code = extracted_codes['carc_codes'][0]
            suggestions.extend([
                f"What RARC codes pair with CARC {code}?",
                f"How to appeal CARC {code}?",
                f"Common causes of CARC {code}?"
            ])
        elif extracted_codes['rarc_codes']:
            code = extracted_codes['rarc_codes'][0]
            suggestions.extend([
                f"What does RARC {code} mean exactly?",
                f"Which CARC codes use RARC {code}?",
                f"How to resolve RARC {code} issues?"
            ])
        else:
            # General suggestions based on recent history
            recent_topics = []
            for chat in chat_history[-3:]:
                if 'carc' in chat['query'].lower():
                    recent_topics.append('CARC codes')
                elif 'rarc' in chat['query'].lower():
                    recent_topics.append('RARC codes')
            
            if 'CARC codes' in recent_topics:
                suggestions.extend([
                    "Most common CARC denial codes?",
                    "CARC vs RARC - what's the difference?",
                    "Appeals process for CARC denials?"
                ])
            else:
                suggestions.extend([
                    "Common healthcare billing denials?",
                    "Coordination of benefits issues?",
                    "Timely filing requirements?"
                ])
        
        return suggestions[:3]

# Initialize Gemini with better error handling
def initialize_gemini():
    try:
        genai.configure(api_key=config.gemini_api_key)
        print_success("Gemini API initialized")
        return True
    except Exception as e:
        print_error(f"API initialization failed: {e}")
        return False

# Enhanced PDF Processor with Smart Chunking
class SmartPDFProcessor:
    def __init__(self, chunk_size=400, chunk_overlap=60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Healthcare-specific section markers
        self.section_markers = [
            r'CARC\s*\d+',
            r'RARC\s*[A-Z]\d+',
            r'Code\s*\d+',
            r'Reason\s*Code',
            r'Adjustment\s*Code',
            r'Denial\s*Code',
            r'Group\s*Code',
            r'Remark\s*Code'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            print_info(f"Extracting text from {pdf_path}...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Better text cleaning
                    page_text = re.sub(r'\s+', ' ', page_text.strip())
                    page_text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', page_text)
                    text += page_text + "\n"
                    
                    if i % 10 == 0 and i > 0:
                        print(f"  Processed {i+1} pages...")
                        
            print_success(f"Extracted text from {len(pdf_reader.pages)} pages")
            return text.strip()
        except Exception as e:
            print_error(f"PDF extraction failed: {e}")
            return ""
    
    def create_smart_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create healthcare-aware chunks that preserve code context"""
        print_info("Creating smart healthcare chunks...")
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        current_chunk = ""
        current_metadata = {"codes": [], "section_type": "general", "chunk_id": 0}
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            # Detect healthcare codes in this paragraph
            codes_found = []
            section_type = "general"
            
            for pattern in self.section_markers:
                matches = re.findall(pattern, para, re.IGNORECASE)
                if matches:
                    codes_found.extend(matches)
                    section_type = "code_definition"
            
            # Check if we should start a new chunk
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            # Smart chunking logic
            should_split = False
            if len(potential_chunk) > self.chunk_size:
                # Don't split if this paragraph contains important codes
                if codes_found and len(current_chunk) > 100:
                    should_split = True
                elif not codes_found and len(potential_chunk) > self.chunk_size * 1.2:
                    should_split = True
            
            if should_split:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "metadata": {
                            **current_metadata,
                            "chunk_id": chunk_id,
                            "length": len(current_chunk)
                        }
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap if needed
                if codes_found:
                    current_chunk = para
                    current_metadata = {
                        "codes": codes_found,
                        "section_type": section_type,
                        "has_overlap": False
                    }
                else:
                    # Include some overlap from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                    current_chunk = overlap_text + "\n\n" + para
                    current_metadata = {
                        "codes": codes_found,
                        "section_type": section_type,
                        "has_overlap": bool(overlap_text)
                    }
            else:
                current_chunk = potential_chunk
                current_metadata["codes"].extend(codes_found)
                if section_type != "general":
                    current_metadata["section_type"] = section_type
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": {
                    **current_metadata,
                    "chunk_id": chunk_id,
                    "length": len(current_chunk)
                }
            })
        
        # Filter out very short chunks
        filtered_chunks = [chunk for chunk in chunks if len(chunk["text"]) > 50]
        print_success(f"Created {len(filtered_chunks)} smart chunks")
        
        return filtered_chunks

# Vector Store
class EnhancedVectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        print_info(f"Loading embedding model: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.texts = []
            self.metadata = []
            print_success("Enhanced vector store initialized")
        except Exception as e:
            print_error(f"Vector store initialization failed: {e}")
            raise
        
    def add_texts(self, chunk_data: List[Dict[str, Any]]):
        """Add texts with enhanced metadata"""
        texts = [chunk["text"] for chunk in chunk_data]
        metadata = [chunk["metadata"] for chunk in chunk_data]
        
        print_info(f"Encoding {len(texts)} text chunks...")
        
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            embeddings = np.array(embeddings)
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings.astype('float32'))
            self.texts.extend(texts)
            self.metadata.extend(metadata)
            
            print_success(f"Added {len(texts)} texts to enhanced vector store")
        except Exception as e:
            print_error(f"Failed to add texts: {e}")
            raise
    
    def enhanced_similarity_search(self, query: str, k: int = 5, code_boost: bool = True) -> List[Dict]:
        """Enhanced search with code-aware ranking"""
        if len(self.texts) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)
        
        # Get more results initially for re-ranking
        search_k = min(k * 2, len(self.texts))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts) and score > config.similarity_threshold:
                result = {
                    "text": self.texts[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx]
                }
                
                # Boost scores for chunks containing relevant codes
                if code_boost and self._contains_query_codes(query, result):
                    result["score"] = min(result["score"] * 1.2, 1.0)
                    result["boosted"] = True
                else:
                    result["boosted"] = False
                
                results.append(result)
        
        # Sort by enhanced scores and return top k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        return results
    
    def _contains_query_codes(self, query: str, result: Dict) -> bool:
        """Check if result contains codes mentioned in query"""
        extractor = HealthcareCodeExtractor()
        query_codes = extractor.extract_healthcare_codes(query)
        
        # Check if any extracted codes appear in the chunk
        chunk_text = result["text"].lower()
        chunk_metadata = result["metadata"]
        
        # Check metadata codes
        if chunk_metadata.get("codes"):
            for code in chunk_metadata["codes"]:
                for code_list in query_codes.values():
                    if any(qcode in code.lower() for qcode in code_list):
                        return True
        
        # Check text content
        for code_list in query_codes.values():
            for code in code_list:
                if code.lower() in chunk_text:
                    return True
        
        return False

# Confidence Scoring
class AdvancedConfidenceScorer:
    """
    Calculate more accurate confidence scores based on multiple factors
    """
    
    def __init__(self):
        self.confidence_factors = {
            'exact_code_match': 0.3,
            'context_relevance': 0.25,
            'completeness': 0.2,
            'specificity': 0.15,
            'source_quality': 0.1
        }
    
    def calculate_confidence(self, query: str, response: str, 
                           retrieved_docs: List[Dict], 
                           extracted_codes: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate enhanced confidence with breakdown"""
        confidence_breakdown = {}
        
        # Factor 1: Exact code matches
        code_match_score = self._calculate_code_match_score(query, response, extracted_codes)
        confidence_breakdown['exact_code_match'] = code_match_score
        
        # Factor 2: Context relevance
        context_score = self._calculate_context_relevance(retrieved_docs)
        confidence_breakdown['context_relevance'] = context_score
        
        # Factor 3: Response completeness
        completeness_score = self._calculate_completeness(response)
        confidence_breakdown['completeness'] = completeness_score
        
        # Factor 4: Specificity
        specificity_score = self._calculate_specificity(response)
        confidence_breakdown['specificity'] = specificity_score
        
        # Factor 5: Source quality
        source_quality_score = self._calculate_source_quality(retrieved_docs)
        confidence_breakdown['source_quality'] = source_quality_score
        
        # Calculate weighted confidence
        total_confidence = sum(
            confidence_breakdown[factor] * weight 
            for factor, weight in self.confidence_factors.items()
        )
        
        return min(total_confidence, 1.0), confidence_breakdown
    
    def _calculate_code_match_score(self, query: str, response: str, extracted_codes: Dict) -> float:
        """Calculate how well response matches query codes"""
        if not any(extracted_codes.values()):
            return 0.6  # Default for non-code queries
        
        query_codes = self._extract_all_codes(query)
        response_codes = self._extract_all_codes(response)
        
        if not query_codes:
            return 0.6
        
        matches = len(set(query_codes) & set(response_codes))
        return min(matches / len(query_codes), 1.0)
    
    def _calculate_context_relevance(self, retrieved_docs: List[Dict]) -> float:
        """Calculate average relevance of retrieved documents"""
        if not retrieved_docs:
            return 0.3
        
        avg_score = np.mean([doc.get('score', 0.5) for doc in retrieved_docs])
        boosted_docs = sum(1 for doc in retrieved_docs if doc.get('boosted', False))
        boost_factor = 1 + (boosted_docs / len(retrieved_docs)) * 0.2
        
        return min(avg_score * boost_factor, 1.0)
    
    def _calculate_completeness(self, response: str) -> float:
        """Calculate response completeness"""
        # Ideal response length range
        ideal_min, ideal_max = 150, 500
        length = len(response)
        
        if length < ideal_min:
            return length / ideal_min * 0.8
        elif length > ideal_max:
            return max(0.7, 1.0 - (length - ideal_max) / 1000)
        else:
            return 1.0
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate how specific and actionable the response is"""
        specific_terms = [
            'denial', 'appeal', 'resubmit', 'documentation', 'medical necessity',
            'prior authorization', 'EOB', 'remittance', 'adjustment', 'coordination',
            'deductible', 'copay', 'patient responsibility', 'contractual',
            'timely filing', 'duplicate', 'provider', 'payer'
        ]
        
        action_terms = [
            'check', 'verify', 'review', 'contact', 'submit', 'correct',
            'appeal', 'resubmit', 'document', 'confirm', 'investigate'
        ]
        
        response_lower = response.lower()
        specific_count = sum(1 for term in specific_terms if term in response_lower)
        action_count = sum(1 for term in action_terms if term in response_lower)
        
        specificity = (specific_count / len(specific_terms)) * 0.7
        actionability = (action_count / len(action_terms)) * 0.3
        
        return min(specificity + actionability, 1.0)
    
    def _calculate_source_quality(self, retrieved_docs: List[Dict]) -> float:
        """Calculate quality of source documents"""
        if not retrieved_docs:
            return 0.3
        
        # Factor in number of sources and their metadata
        source_count_score = min(len(retrieved_docs) / 3, 1.0)
        
        # Bonus for code-definition chunks
        code_chunks = sum(1 for doc in retrieved_docs 
                         if doc.get('metadata', {}).get('section_type') == 'code_definition')
        code_bonus = (code_chunks / len(retrieved_docs)) * 0.3
        
        return min(source_count_score + code_bonus, 1.0)
    
    def _extract_all_codes(self, text: str) -> List[str]:
        """Extract all healthcare codes from text"""
        patterns = [
            r'CARC\s*\d+', r'RARC\s*[A-Z]\d+', r'CO[-\s]*\d+',
            r'PR[-\s]*\d+', r'OA[-\s]*\d+', r'PI[-\s]*\d+',
            r'CR[-\s]*\d+', r'\b\d{1,3}\b'
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            codes.extend(matches)
        
        return list(set(codes))  # Remove duplicates

# RAG Chatbot
class EnhancedTerminalRAGChatbot:
    def __init__(self, config: EnhancedRAGConfig, vector_store: EnhancedVectorStore):
        self.config = config
        self.vector_store = vector_store
        
        # Initialize enhancement components
        self.code_extractor = HealthcareCodeExtractor()
        self.confidence_scorer = AdvancedConfidenceScorer()
        
        self.generation_config = genai.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        self.model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.chat_history = []
    
    def generate_enhanced_response(self, query: str) -> Dict[str, Any]:
        try:
            print(f"{Colors.OKCYAN}🧠 Processing query...{Colors.ENDC}", end="", flush=True)
            
            # Extract healthcare codes from query
            extracted_codes = self.code_extractor.extract_healthcare_codes(query)
            
            # Enhance query with code context
            enhanced_query = self.code_extractor.enhance_query_with_codes(query, extracted_codes)
            
            # Get relevant documents with enhanced search
            retrieved_docs = self.vector_store.enhanced_similarity_search(
                enhanced_query, k=config.top_k_results, code_boost=True
            )
            
            print("\r" + " " * 25 + "\r", end="", flush=True)
            
            # Create enhanced prompt
            if not retrieved_docs:
                prompt = f"""
{HEALTHCARE_CLAIMS_SPECIALIST_PROMPT}

Query: {query}

Please respond based on your healthcare claims expertise. Be specific and actionable.
"""
            else:
                # Prioritize code-definition chunks
                context_chunks = []
                for doc in retrieved_docs:
                    chunk_info = f"Source: {doc['metadata'].get('section_type', 'general')}"
                    if doc['metadata'].get('codes'):
                        chunk_info += f" (Contains: {', '.join(doc['metadata']['codes'])})"
                    context_chunks.append(f"{chunk_info}\nContent: {doc['text']}\n")
                
                context = "\n---\n".join(context_chunks)
                
                prompt = f"""
{HEALTHCARE_CLAIMS_SPECIALIST_PROMPT}

Knowledge Base Context:
{context}

Query: {query}

Please provide a comprehensive response using both the context and your expertise. 
Focus on actionable advice and specific next steps.
"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            response_text = response.text.strip() if hasattr(response, 'text') else "Unable to generate response."
            
            # Calculate enhanced confidence
            confidence, confidence_breakdown = self.confidence_scorer.calculate_confidence(
                query, response_text, retrieved_docs, extracted_codes
            )
            
            # Generate suggestions
            suggestions = self.code_extractor.get_query_suggestions(query, self.chat_history)
            
            result = {
                "response": response_text,
                "confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "sources": len(retrieved_docs),
                "extracted_codes": extracted_codes,
                "suggestions": suggestions,
                "enhanced_query": enhanced_query != query,
                "boosted_sources": sum(1 for doc in retrieved_docs if doc.get('boosted', False))
            }
            
            # chat history
            self.chat_history.append({
                "query": query,
                "response": response_text,
                "confidence": confidence,
                "extracted_codes": extracted_codes
            })
            
            return result
            
        except Exception as e:
            print_error(f"Generation error: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0,
                "sources": 0,
                "extracted_codes": {},
                "suggestions": []
            }
    
    def start_enhanced_terminal_chat(self):
        """Enhanced terminal chat interface with new features"""
        print_header("Enhanced Healthcare Claims Specialist")
        print_info("Ask questions about CARC/RARC codes, healthcare billing, EDI, etc.")
        print_info("New features: Code extraction, smart suggestions, confidence scoring")
        print(f"{Colors.WARNING}Commands: 'quit', 'exit', 'help', 'history', 'clear', 'stats'{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        while True:
            try:
                print(f"\n{Colors.BOLD}{Colors.OKGREEN}You:{Colors.ENDC} ", end="")
                query = input().strip()
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._goodbye()
                    break
                elif query.lower() == 'help':
                    self._show_enhanced_help()
                    continue
                elif query.lower() == 'history':
                    self._show_history()
                    continue
                elif query.lower() == 'stats':
                    self._show_stats()
                    continue
                elif query.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print_header("Enhanced Healthcare Claims Specialist")
                    continue
                elif not query:
                    continue
                
                # Process the query
                start_time = time.time()
                result = self.generate_enhanced_response(query)
                processing_time = time.time() - start_time
                
                # Display response with enhanced formatting
                self._display_enhanced_response(result, processing_time)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Chat interrupted. Type 'quit' to exit properly.{Colors.ENDC}")
            except Exception as e:
                print_error(f"Unexpected error: {e}")
    
    def _display_enhanced_response(self, result: Dict[str, Any], processing_time: float):
        """Display response with enhanced formatting and confidence indicators"""
        response = result["response"]
        confidence = result["confidence"]
        sources = result["sources"]
        extracted_codes = result["extracted_codes"]
        suggestions = result["suggestions"]
        
        # Display main response
        print(f"\n{Colors.BOLD}{Colors.PURPLE}Healthcare Specialist:{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        # Highlight codes in response if enabled
        if config.highlight_codes and any(extracted_codes.values()):
            highlighted_response = self._highlight_codes_in_text(response, extracted_codes)
            print(highlighted_response)
        else:
            print(response)
        
        # Display metadata
        print(f"\n{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        # Confidence indicator
        confidence_color = Colors.OKGREEN if confidence > 0.7 else Colors.WARNING if confidence > 0.4 else Colors.FAIL
        confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        print(f"{confidence_emoji} Confidence: {confidence_color}{confidence:.1%}{Colors.ENDC}")
        
        # Show confidence warning if low
        if confidence < config.confidence_threshold_warning:
            print_warning("Low confidence - consider asking for more specific information")
        
        # Sources and processing info
        info_items = [
            f"Sources: {sources}",
            f"Processing: {processing_time:.2f}s"
        ]
        
        if result.get("boosted_sources", 0) > 0:
            info_items.append(f"Code-matched: {result['boosted_sources']}")
        
        if result.get("enhanced_query", False):
            info_items.append("Query enhanced")
        
        print(f"{Colors.OKCYAN}📊 {' | '.join(info_items)}{Colors.ENDC}")
        
        # Show extracted codes if any
        if any(extracted_codes.values()):
            code_info = []
            for code_type, codes in extracted_codes.items():
                if codes:
                    code_info.append(f"{code_type.replace('_codes', '').upper()}: {', '.join(codes)}")
            
            if code_info:
                print(f"{Colors.YELLOW}🏷️  Detected codes: {' | '.join(code_info)}{Colors.ENDC}")
        
        # Show suggestions if enabled
        if config.include_suggestions and suggestions:
            print(f"\n{Colors.OKCYAN}💡 Related questions you might ask:{Colors.ENDC}")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{Colors.OKCYAN}   {i}. {suggestion}{Colors.ENDC}")
    
    def _highlight_codes_in_text(self, text: str, extracted_codes: Dict) -> str:
        """Highlight healthcare codes in the response text"""
        highlighted_text = text
        
        # Collect all codes to highlight
        all_codes = []
        for codes_list in extracted_codes.values():
            all_codes.extend(codes_list)
        
        # Highlight each code
        for code in all_codes:
            pattern = re.compile(re.escape(code), re.IGNORECASE)
            highlighted_text = pattern.sub(f"{Colors.BOLD}{Colors.YELLOW}{code}{Colors.ENDC}", highlighted_text)
        
        return highlighted_text
    
    def _show_enhanced_help(self):
        """Display enhanced help information"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}Enhanced Healthcare Claims Specialist - Help{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        print(f"{Colors.BOLD}Available Commands:{Colors.ENDC}")
        commands = [
            ("help", "Show this help message"),
            ("history", "Show recent chat history"),
            ("stats", "Show system statistics"),
            ("clear", "Clear the terminal screen"),
            ("quit/exit/q", "Exit the application")
        ]
        
        for cmd, desc in commands:
            print(f"  {Colors.OKCYAN}{cmd:<12}{Colors.ENDC} - {desc}")
        
        print(f"\n{Colors.BOLD}Enhanced Features:{Colors.ENDC}")
        features = [
            "🔍 Automatic healthcare code detection (CARC, RARC, CO, PR, etc.)",
            "🎯 Smart query enhancement for better results",
            "📊 Confidence scoring with detailed breakdown",
            "💡 Contextual suggestions for follow-up questions",
            "🏷️  Code highlighting in responses",
            "⚡ Boosted search results for code-specific queries"
        ]
        
        for feature in features:
            print(f"  {feature}")
        
        print(f"\n{Colors.BOLD}Example Queries:{Colors.ENDC}")
        examples = [
            "What does CARC 1 mean?",
            "Difference between CARC 27 and CARC 29?",
            "How to appeal RARC N115?",
            "CO 45 explanation and next steps",
            "Patient responsibility codes vs contractual adjustments"
        ]
        
        for example in examples:
            print(f"  {Colors.OKGREEN}• {example}{Colors.ENDC}")
    
    def _show_history(self):
        """Display recent chat history"""
        if not self.chat_history:
            print_info("No chat history available")
            return
        
        print(f"\n{Colors.BOLD}{Colors.HEADER}Recent Chat History{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        # Show last 5 interactions
        recent_history = self.chat_history[-5:]
        
        for i, chat in enumerate(recent_history, 1):
            confidence_emoji = "🟢" if chat['confidence'] > 0.7 else "🟡" if chat['confidence'] > 0.4 else "🔴"
            
            print(f"\n{Colors.BOLD}{i}. Query:{Colors.ENDC} {chat['query'][:80]}{'...' if len(chat['query']) > 80 else ''}")
            print(f"   {confidence_emoji} Confidence: {chat['confidence']:.1%}")
            
            if any(chat['extracted_codes'].values()):
                codes = []
                for _, code_list in chat['extracted_codes'].items():
                    if code_list:
                        codes.extend(code_list)
                print(f"   🏷️  Codes: {', '.join(codes)}")
    
    def _show_stats(self):
        """Display system statistics"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}System Statistics{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        # Chat statistics
        total_queries = len(self.chat_history)
        if total_queries > 0:
            avg_confidence = np.mean([chat['confidence'] for chat in self.chat_history])
            high_confidence_queries = sum(1 for chat in self.chat_history if chat['confidence'] > 0.7)
            
            print(f"{Colors.BOLD}Chat Statistics:{Colors.ENDC}")
            print(f"  Total queries: {total_queries}")
            print(f"  Average confidence: {avg_confidence:.1%}")
            print(f"  High confidence queries: {high_confidence_queries} ({high_confidence_queries/total_queries:.1%})")
        
        # Vector store statistics
        print(f"\n{Colors.BOLD}Knowledge Base:{Colors.ENDC}")
        print(f"  Total documents: {len(self.vector_store.texts)}")
        print(f"  Embedding dimension: {self.vector_store.embedding_dim}")
        print(f"  Search threshold: {config.similarity_threshold}")
        
        # Code extraction statistics
        if total_queries > 0:
            code_queries = sum(1 for chat in self.chat_history 
                             if any(chat['extracted_codes'].values()))
            
            print(f"\n{Colors.BOLD}Code Detection:{Colors.ENDC}")
            print(f"  Queries with codes: {code_queries} ({code_queries/total_queries:.1%})")
            
            # Most common code types
            code_types = {}
            for chat in self.chat_history:
                for code_type, codes in chat['extracted_codes'].items():
                    if codes:
                        code_types[code_type] = code_types.get(code_type, 0) + len(codes)
            
            if code_types:
                print("  Most detected codes:")
                for code_type, count in sorted(code_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    {code_type.replace('_codes', '').upper()}: {count}")
    
    def _goodbye(self):
        """Display goodbye message with session summary"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}Session Summary{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'─' * 60}{Colors.ENDC}")
        
        if self.chat_history:
            total_queries = len(self.chat_history)
            avg_confidence = np.mean([chat['confidence'] for chat in self.chat_history])
            
            print(f"📊 Processed {total_queries} queries")
            print(f"🎯 Average confidence: {avg_confidence:.1%}")
            print(f"💡 Enhanced healthcare code assistance provided")
        else:
            print("👋 Thanks for trying the Enhanced Healthcare Claims Specialist!")
        
        print(f"\n{Colors.OKGREEN}Thank you for using the Enhanced Healthcare Claims Specialist!{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Stay informed, stay compliant! 🏥✨{Colors.ENDC}")

# Main execution function
def main():
    """Main function to run the enhanced RAG system"""
    print_header("Enhanced Terminal RAG Chat System")
    print_info("Initializing enhanced healthcare claims specialist...")
    
    # Initialize Gemini
    if not initialize_gemini():
        print_error("Failed to initialize Gemini API. Please check your API key.")
        return
    
    # Initialize components
    try:
        # Create enhanced vector store
        print_info("Setting up enhanced vector store...")
        vector_store = EnhancedVectorStore(config.embedding_model)
        
        # Initialize PDF processor
        pdf_processor = SmartPDFProcessor(config.chunk_size, config.chunk_overlap)
        
        # Check for PDF files
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        
        if pdf_files:
            print_success(f"Found {len(pdf_files)} PDF files")
            
            all_chunks = []
            for pdf_file in pdf_files:
                print_info(f"Processing {pdf_file}...")
                text = pdf_processor.extract_text_from_pdf(pdf_file)
                if text:
                    chunks = pdf_processor.create_smart_chunks(text)
                    all_chunks.extend(chunks)
                    print_success(f"Created {len(chunks)} chunks from {pdf_file}")
            
            if all_chunks:
                print_info("Adding chunks to enhanced vector store...")
                vector_store.add_texts(all_chunks)
                print_success(f"Knowledge base loaded with {len(all_chunks)} total chunks")
            else:
                print_warning("No valid content extracted from PDFs")
        else:
            print_warning("No PDF files found. Running with base knowledge only.")
        
        # Initialize enhanced chatbot
        chatbot = EnhancedTerminalRAGChatbot(config, vector_store)
        
        # Start enhanced chat
        chatbot.start_enhanced_terminal_chat()
        
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return

if __name__ == "__main__":
    main()
                    
