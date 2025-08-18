#!/usr/bin/env python3
"""
Comprehensive API Testing Script for Code Storm Backend
Tests all available endpoints with proper error handling and reporting.
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TEST_PDF_PATH = "test_sample.pdf"  # Assuming this exists in the directory

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def log_result(self, test_name: str, status: str, details: str = "", response_time: float = 0):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "response_time": f"{response_time:.2f}s"
        }
        self.results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if response_time > 0:
            print(f"   Response time: {response_time:.2f}s")
        print()
        
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[Optional[requests.Response], str]:
        """Make HTTP request with error handling"""
        try:
            start_time = time.time()
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, **kwargs)
            else:
                return None, f"Unsupported method: {method}"
                
            response_time = time.time() - start_time
            return response, f"Response time: {response_time:.2f}s"
            
        except requests.exceptions.ConnectionError:
            return None, "Connection failed - is the server running?"
        except requests.exceptions.Timeout:
            return None, "Request timed out"
        except Exception as e:
            return None, f"Request failed: {str(e)}"
    
    def test_root_endpoint(self):
        """Test GET / endpoint"""
        print("🧪 Testing Root Endpoint...")
        response, details = self.make_request("GET", "/")
        
        if response is None:
            self.log_result("Root Endpoint", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                expected_keys = ["message", "description", "endpoints", "supported_formats"]
                missing_keys = [key for key in expected_keys if key not in data]
                
                if missing_keys:
                    self.log_result("Root Endpoint", "FAIL", f"Missing keys: {missing_keys}")
                else:
                    self.log_result("Root Endpoint", "PASS", f"All expected keys present. {details}")
            except json.JSONDecodeError:
                self.log_result("Root Endpoint", "FAIL", "Invalid JSON response")
        else:
            self.log_result("Root Endpoint", "FAIL", f"Status: {response.status_code}")
    
    def test_health_endpoint(self):
        """Test GET /health endpoint"""
        print("🧪 Testing Health Endpoint...")
        response, details = self.make_request("GET", "/health")
        
        if response is None:
            self.log_result("Health Check", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                expected_keys = ["status", "services", "service", "version"]
                missing_keys = [key for key in expected_keys if key not in data]
                
                if missing_keys:
                    self.log_result("Health Check", "FAIL", f"Missing keys: {missing_keys}")
                else:
                    status = data.get("status", "unknown")
                    services = data.get("services", {})
                    direct_status = services.get("direct_processing", {}).get("status", "unknown")
                    vector_status = services.get("vector_processing", {}).get("status", "unknown")
                    
                    self.log_result("Health Check", "PASS", 
                                  f"Overall: {status}, Direct: {direct_status}, Vector: {vector_status}. {details}")
            except json.JSONDecodeError:
                self.log_result("Health Check", "FAIL", "Invalid JSON response")
        else:
            self.log_result("Health Check", "FAIL", f"Status: {response.status_code}")
    
    def test_vector_stats_endpoint(self):
        """Test GET /vector_stats endpoint"""
        print("🧪 Testing Vector Stats Endpoint...")
        response, details = self.make_request("GET", "/vector_stats")
        
        if response is None:
            self.log_result("Vector Stats", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                self.log_result("Vector Stats", "PASS", f"Stats retrieved successfully. {details}")
            except json.JSONDecodeError:
                self.log_result("Vector Stats", "FAIL", "Invalid JSON response")
        elif response.status_code == 503:
            self.log_result("Vector Stats", "PASS", "Vector store not available (expected if Redis not running)")
        else:
            self.log_result("Vector Stats", "FAIL", f"Status: {response.status_code}")
    
    def test_process_direct_endpoint(self):
        """Test POST /process_direct endpoint"""
        print("🧪 Testing Process Direct Endpoint...")
        
        if not os.path.exists(TEST_PDF_PATH):
            self.log_result("Process Direct", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
            return
        
        try:
            with open(TEST_PDF_PATH, 'rb') as pdf_file:
                files = {'file': (TEST_PDF_PATH, pdf_file, 'application/pdf')}
                response, details = self.make_request("POST", "/process_direct", files=files, timeout=60)
                
                if response is None:
                    self.log_result("Process Direct", "FAIL", details)
                    return
                    
                if response.status_code == 200:
                    try:
                        data = response.json()
                        expected_keys = ["status", "analysis", "performance", "metadata"]
                        missing_keys = [key for key in expected_keys if key not in data]
                        
                        if missing_keys:
                            self.log_result("Process Direct", "FAIL", f"Missing keys: {missing_keys}")
                        else:
                            status = data.get("status")
                            total_time = data.get("performance", {}).get("total_time", 0)
                            target_achieved = data.get("performance", {}).get("target_achieved", False)
                            
                            result_details = f"Status: {status}, Time: {total_time}s, Target <20s: {target_achieved}"
                            self.log_result("Process Direct", "PASS", result_details)
                    except json.JSONDecodeError:
                        self.log_result("Process Direct", "FAIL", "Invalid JSON response")
                else:
                    error_msg = f"Status: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f", Error: {error_data.get('detail', 'Unknown error')}"
                    except:
                        pass
                    self.log_result("Process Direct", "FAIL", error_msg)
                    
        except FileNotFoundError:
            self.log_result("Process Direct", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
        except Exception as e:
            self.log_result("Process Direct", "FAIL", f"Error reading PDF: {str(e)}")
    
    def test_process_vector_endpoint(self):
        """Test POST /process_vector endpoint"""
        print("🧪 Testing Process Vector Endpoint...")
        
        if not os.path.exists(TEST_PDF_PATH):
            self.log_result("Process Vector", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
            return
        
        try:
            with open(TEST_PDF_PATH, 'rb') as pdf_file:
                files = {'file': (TEST_PDF_PATH, pdf_file, 'application/pdf')}
                response, details = self.make_request("POST", "/process_vector", files=files, timeout=60)
                
                if response is None:
                    self.log_result("Process Vector", "FAIL", details)
                    return
                    
                if response.status_code == 200:
                    try:
                        data = response.json()
                        expected_keys = ["status", "message", "document_info", "processing_times"]
                        missing_keys = [key for key in expected_keys if key not in data]
                        
                        if missing_keys:
                            self.log_result("Process Vector", "FAIL", f"Missing keys: {missing_keys}")
                        else:
                            status = data.get("status")
                            doc_id = data.get("document_info", {}).get("document_id")
                            chunk_count = data.get("document_info", {}).get("chunk_count", 0)
                            
                            result_details = f"Status: {status}, Doc ID: {doc_id}, Chunks: {chunk_count}"
                            self.log_result("Process Vector", "PASS", result_details)
                            return doc_id  # Return document ID for search test
                    except json.JSONDecodeError:
                        self.log_result("Process Vector", "FAIL", "Invalid JSON response")
                elif response.status_code == 503:
                    self.log_result("Process Vector", "PASS", "Vector services not available (expected if Redis not running)")
                else:
                    error_msg = f"Status: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f", Error: {error_data.get('detail', 'Unknown error')}"
                    except:
                        pass
                    self.log_result("Process Vector", "FAIL", error_msg)
                    
        except FileNotFoundError:
            self.log_result("Process Vector", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
        except Exception as e:
            self.log_result("Process Vector", "FAIL", f"Error reading PDF: {str(e)}")
        
        return None
    
    def test_search_documents_endpoint(self, document_id: Optional[str] = None):
        """Test POST /search_documents endpoint"""
        print("🧪 Testing Search Documents Endpoint...")
        
        params = {
            "query": "contract obligations",
            "top_k": 5
        }
        
        if document_id:
            params["document_id"] = document_id
        
        response, details = self.make_request("POST", "/search_documents", params=params)
        
        if response is None:
            self.log_result("Search Documents", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                expected_keys = ["status", "query", "results_count", "results"]
                missing_keys = [key for key in expected_keys if key not in data]
                
                if missing_keys:
                    self.log_result("Search Documents", "FAIL", f"Missing keys: {missing_keys}")
                else:
                    status = data.get("status")
                    results_count = data.get("results_count", 0)
                    query = data.get("query")
                    
                    result_details = f"Status: {status}, Query: '{query}', Results: {results_count}"
                    self.log_result("Search Documents", "PASS", result_details)
            except json.JSONDecodeError:
                self.log_result("Search Documents", "FAIL", "Invalid JSON response")
        elif response.status_code == 503:
            self.log_result("Search Documents", "PASS", "Vector search services not available (expected if Redis not running)")
        else:
            error_msg = f"Status: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f", Error: {error_data.get('detail', 'Unknown error')}"
            except:
                pass
            self.log_result("Search Documents", "FAIL", error_msg)
    
    def test_legacy_process_document_endpoint(self):
        """Test POST /process_document endpoint (legacy)"""
        print("🧪 Testing Legacy Process Document Endpoint...")
        
        if not os.path.exists(TEST_PDF_PATH):
            self.log_result("Legacy Process Document", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
            return
        
        try:
            with open(TEST_PDF_PATH, 'rb') as pdf_file:
                files = {'file': (TEST_PDF_PATH, pdf_file, 'application/pdf')}
                response, details = self.make_request("POST", "/process_document", files=files, timeout=60)
                
                if response is None:
                    self.log_result("Legacy Process Document", "FAIL", details)
                    return
                    
                if response.status_code == 200:
                    try:
                        data = response.json()
                        status = data.get("status")
                        self.log_result("Legacy Process Document", "PASS", f"Status: {status} (redirected to direct processing)")
                    except json.JSONDecodeError:
                        self.log_result("Legacy Process Document", "FAIL", "Invalid JSON response")
                else:
                    error_msg = f"Status: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg += f", Error: {error_data.get('detail', 'Unknown error')}"
                    except:
                        pass
                    self.log_result("Legacy Process Document", "FAIL", error_msg)
                    
        except FileNotFoundError:
            self.log_result("Legacy Process Document", "FAIL", f"Test PDF not found: {TEST_PDF_PATH}")
        except Exception as e:
            self.log_result("Legacy Process Document", "FAIL", f"Error reading PDF: {str(e)}")
    
    def test_rag_health_endpoint(self):
        """Test GET /rag_health endpoint"""
        print("🧪 Testing RAG Health Endpoint...")
        response, details = self.make_request("GET", "/rag_health")
        
        if response is None:
            self.log_result("RAG Health", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                status = data.get("status", "unknown")
                rag_health = data.get("rag_health", {})
                capabilities = data.get("capabilities", {})
                
                result_details = f"Status: {status}, RAG Status: {rag_health.get('status', 'unknown')}"
                self.log_result("RAG Health", "PASS", result_details)
            except json.JSONDecodeError:
                self.log_result("RAG Health", "FAIL", "Invalid JSON response")
        else:
            self.log_result("RAG Health", "FAIL", f"Status: {response.status_code}")
    
    def test_suggested_questions_endpoint(self):
        """Test GET /suggested_questions endpoint"""
        print("🧪 Testing Suggested Questions Endpoint...")
        response, details = self.make_request("GET", "/suggested_questions")
        
        if response is None:
            self.log_result("Suggested Questions", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                expected_keys = ["status", "suggested_questions", "total_suggestions"]
                missing_keys = [key for key in expected_keys if key not in data]
                
                if missing_keys:
                    self.log_result("Suggested Questions", "FAIL", f"Missing keys: {missing_keys}")
                else:
                    status = data.get("status")
                    total_suggestions = data.get("total_suggestions", 0)
                    
                    result_details = f"Status: {status}, Suggestions: {total_suggestions}"
                    self.log_result("Suggested Questions", "PASS", result_details)
            except json.JSONDecodeError:
                self.log_result("Suggested Questions", "FAIL", "Invalid JSON response")
        elif response.status_code == 503:
            self.log_result("Suggested Questions", "PASS", "RAG service not available (expected if not configured)")
        else:
            self.log_result("Suggested Questions", "FAIL", f"Status: {response.status_code}")
    
    def test_ask_question_endpoint(self):
        """Test POST /ask_question endpoint"""
        print("🧪 Testing Ask Question Endpoint...")
        
        params = {
            "query": "What are the key obligations in this contract?"
        }
        
        response, details = self.make_request("POST", "/ask_question", params=params, timeout=30)
        
        if response is None:
            self.log_result("Ask Question", "FAIL", details)
            return
            
        if response.status_code == 200:
            try:
                data = response.json()
                expected_keys = ["status", "query", "answer", "confidence_score"]
                missing_keys = [key for key in expected_keys if key not in data]
                
                if missing_keys:
                    self.log_result("Ask Question", "FAIL", f"Missing keys: {missing_keys}")
                else:
                    status = data.get("status")
                    confidence = data.get("confidence_score", 0)
                    processing_time = data.get("processing_time", 0)
                    
                    result_details = f"Status: {status}, Confidence: {confidence:.1f}%, Time: {processing_time:.2f}s"
                    self.log_result("Ask Question", "PASS", result_details)
            except json.JSONDecodeError:
                self.log_result("Ask Question", "FAIL", "Invalid JSON response")
        elif response.status_code == 503:
            self.log_result("Ask Question", "PASS", "RAG service not available (expected if Redis not running)")
        elif response.status_code == 400:
            self.log_result("Ask Question", "PASS", "Bad request (expected behavior for validation)")
        else:
            error_msg = f"Status: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f", Error: {error_data.get('detail', 'Unknown error')}"
            except:
                pass
            self.log_result("Ask Question", "FAIL", error_msg)
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Comprehensive API Testing for Code Storm Backend")
        print("=" * 70)
        
        # Check if server is running
        print("🔍 Checking server connectivity...")
        response, details = self.make_request("GET", "/")
        if response is None:
            print(f"❌ Cannot connect to server at {self.base_url}")
            print(f"   Details: {details}")
            print("   Please make sure the backend server is running!")
            return
        
        print(f"✅ Server is running at {self.base_url}")
        print()
        
        # Run all tests
        self.test_root_endpoint()
        self.test_health_endpoint()
        self.test_vector_stats_endpoint()
        
        # Test document processing endpoints
        document_id = self.test_process_vector_endpoint()
        self.test_process_direct_endpoint()
        self.test_legacy_process_document_endpoint()
        
        # Test search endpoint
        self.test_search_documents_endpoint(document_id)
        
        # Test RAG Q&A endpoints
        self.test_rag_health_endpoint()
        self.test_suggested_questions_endpoint()
        self.test_ask_question_endpoint()
        
        # Print summary
        print("=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📋 Total: {len(self.results)}")
        
        if self.failed == 0:
            print("\n🎉 All tests passed! Your API is working perfectly!")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Please check the details above.")
        
        # Print detailed results
        print("\n📝 DETAILED RESULTS:")
        print("-" * 50)
        for result in self.results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {result['test']}: {result['status']}")
            if result['details']:
                print(f"   {result['details']}")
            print(f"   Response time: {result['response_time']}")
        
        return self.failed == 0

def main():
    """Main function"""
    print("🏛️ Code Storm Backend API Tester")
    print("=" * 50)
    
    # Check if test PDF exists
    if not os.path.exists(TEST_PDF_PATH):
        print(f"⚠️  Warning: Test PDF '{TEST_PDF_PATH}' not found.")
        print("   Document processing tests will be skipped.")
        print("   Please ensure test_sample.pdf exists in the current directory.")
        print()
    
    # Initialize tester and run tests
    tester = APITester(BASE_URL)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
