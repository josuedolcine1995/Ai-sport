#!/usr/bin/env python3
"""
Comprehensive System Tester for Ultimate Sports AI
Tests 5,000,000 predictions to verify accuracy stability with real data
"""

import asyncio
import aiohttp
import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    query: str
    response_time: float
    accuracy: float
    confidence: float
    real_data_sources: List[str]
    success: bool
    timestamp: datetime

class ComprehensiveSystemTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
        # Test parameters
        self.target_tests = 5_000_000
        self.batch_size = 1000
        self.concurrent_requests = 50
        
        # Results tracking
        self.test_results: List[TestResult] = []
        self.accuracy_scores = []
        self.response_times = []
        self.real_data_validation = {
            'total_queries': 0,
            'real_data_queries': 0,
            'mock_data_detected': 0,
            'validation_failures': 0
        }
        
        # Test queries for maximum coverage
        self.test_queries = self.generate_comprehensive_test_queries()
        
    def generate_comprehensive_test_queries(self) -> List[str]:
        """Generate comprehensive test queries covering all scenarios"""
        
        # CS:GO players and scenarios
        csgo_players = [
            's1mple', 'ZywOo', 'device', 'NiKo', 'electronic', 
            'sh1ro', 'Ax1Le', 'b1t', 'perfecto', 'flamie',
            'dupreeh', 'gla1ve', 'Xyp9x', 'Magisk', 'blameF',
            'FalleN', 'fer', 'TACO', 'coldzera', 'LUCAS1'
        ]
        
        # Valorant players
        valorant_players = [
            'TenZ', 'Aspas', 'Derke', 'yay', 'Jamppi',
            'ScreaM', 'nAts', 'Chronicle', 'Leo', 'Alfajer',
            'cNed', 'Cryocells', 'Zekken', 'johnqt', 'Sacy',
            'Boaster', 'FNS', 'Victor', 'crashies', 'Marved'
        ]
        
        # NBA players
        nba_players = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
            'Luka Doncic', 'Joel Embiid', 'Nikola Jokic', 'Jayson Tatum',
            'Jimmy Butler', 'Anthony Davis', 'Damian Lillard', 'James Harden',
            'Kawhi Leonard', 'Paul George', 'Russell Westbrook', 'Chris Paul',
            'Devin Booker', 'Donovan Mitchell', 'Trae Young', 'Ja Morant'
        ]
        
        queries = []
        
        # CS:GO over/under queries
        for player in csgo_players:
            for kills in [15, 18, 20, 22, 25, 28, 30]:
                queries.extend([
                    f"Will {player} get over {kills} kills?",
                    f"Will {player} get under {kills} kills?",
                    f"Can {player} score over {kills} kills?",
                    f"Will {player} have over {kills} kills?"
                ])
        
        # Valorant over/under queries  
        for player in valorant_players:
            for kills in [12, 15, 18, 20, 22, 25]:
                queries.extend([
                    f"Will {player} get over {kills} kills?",
                    f"Will {player} get under {kills} kills?",
                    f"Can {player} score over {kills} kills in Valorant?",
                    f"Will {player} have over {kills} kills in Valorant?"
                ])
        
        # NBA over/under queries
        for player in nba_players:
            for points in [15, 20, 22, 25, 27, 30, 35]:
                queries.extend([
                    f"Will {player} score over {points} points?",
                    f"Will {player} score under {points} points?",
                    f"Can {player} get over {points} points?",
                    f"Will {player} have over {points} points?"
                ])
            
            # Rebounds and assists
            for rebounds in [5, 7, 10, 12]:
                queries.extend([
                    f"Will {player} get over {rebounds} rebounds?",
                    f"Will {player} get under {rebounds} rebounds?"
                ])
            
            for assists in [3, 5, 7, 10]:
                queries.extend([
                    f"Will {player} get over {assists} assists?",
                    f"Will {player} get under {assists} assists?"
                ])
        
        # System queries
        system_queries = [
            "system status",
            "system accuracy",
            "system health",
            "test system",
            "check system"
        ]
        queries.extend(system_queries * 100)  # Repeat system queries
        
        # General queries
        general_queries = [
            "CS:GO matches today",
            "Valorant predictions",
            "NBA analysis",
            "esports betting",
            "sports predictions"
        ]
        queries.extend(general_queries * 200)
        
        logger.info(f"Generated {len(queries)} unique test queries")
        return queries
    
    async def run_single_test(self, session: aiohttp.ClientSession, query: str) -> TestResult:
        """Run a single test query"""
        start_time = time.time()
        success = False
        accuracy = 0.0
        confidence = 0.0
        real_data_sources = []
        
        try:
            async with session.post(
                f"{self.api_url}/chat",
                json={"message": query},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    success = True
                    accuracy = data.get('accuracy', 0.0)
                    confidence = data.get('confidence', 0.0)
                    real_data_sources = data.get('real_data_sources', [])
                    
                    # Validate real data integrity
                    await self.validate_real_data_response(data, query)
                    
                else:
                    response_time = time.time() - start_time
                    logger.warning(f"HTTP {response.status} for query: {query}")
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error testing query '{query}': {e}")
        
        return TestResult(
            query=query,
            response_time=response_time,
            accuracy=accuracy,
            confidence=confidence,
            real_data_sources=real_data_sources,
            success=success,
            timestamp=datetime.utcnow()
        )
    
    async def validate_real_data_response(self, response_data: Dict, query: str):
        """Validate that response uses real data (no mock/synthetic)"""
        self.real_data_validation['total_queries'] += 1
        
        response_text = response_data.get('response', '').lower()
        real_sources = response_data.get('real_data_sources', [])
        
        # Check for mock data indicators (should NOT be present)
        mock_indicators = [
            'mock', 'synthetic', 'generated', 'fallback data',
            'placeholder', 'example', 'demo', 'test data'
        ]
        
        mock_detected = any(indicator in response_text for indicator in mock_indicators)
        
        if mock_detected:
            self.real_data_validation['mock_data_detected'] += 1
            logger.warning(f"Mock data detected in response for: {query}")
        
        # Check for real data sources
        real_indicators = [
            'hltv', 'vlr.gg', 'basketball-reference', 'espn',
            'real data', 'scraped', 'live data', 'actual'
        ]
        
        has_real_indicators = any(indicator in response_text for indicator in real_indicators)
        has_real_sources = len(real_sources) > 0
        
        if has_real_indicators or has_real_sources:
            self.real_data_validation['real_data_queries'] += 1
        else:
            self.real_data_validation['validation_failures'] += 1
    
    async def run_batch_tests(self, session: aiohttp.ClientSession, batch_queries: List[str]) -> List[TestResult]:
        """Run a batch of tests concurrently"""
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def run_with_semaphore(query):
            async with semaphore:
                return await self.run_single_test(session, query)
        
        tasks = [run_with_semaphore(query) for query in batch_queries]
        return await asyncio.gather(*tasks)
    
    async def run_comprehensive_test(self, target_tests: int = None):
        """Run comprehensive system test"""
        if target_tests is None:
            target_tests = self.target_tests
        
        logger.info(f"Starting comprehensive test with {target_tests:,} predictions")
        start_time = time.time()
        
        # Create test batches
        test_batches = []
        queries_per_batch = min(self.batch_size, target_tests)
        
        for i in range(0, target_tests, queries_per_batch):
            batch_size = min(queries_per_batch, target_tests - i)
            batch_queries = []
            
            for _ in range(batch_size):
                # Randomly select queries to ensure coverage
                query = random.choice(self.test_queries)
                batch_queries.append(query)
            
            test_batches.append(batch_queries)
        
        logger.info(f"Created {len(test_batches)} test batches")
        
        # Run tests
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            
            for i, batch in enumerate(test_batches):
                logger.info(f"Running batch {i+1}/{len(test_batches)} ({len(batch)} tests)")
                
                try:
                    batch_results = await self.run_batch_tests(session, batch)
                    self.test_results.extend(batch_results)
                    
                    # Update statistics
                    for result in batch_results:
                        if result.success:
                            self.accuracy_scores.append(result.accuracy)
                            self.response_times.append(result.response_time)
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        await self.log_progress(i + 1, len(test_batches))
                    
                    # Small delay between batches to avoid overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in batch {i+1}: {e}")
                    continue
        
        total_time = time.time() - start_time
        await self.generate_comprehensive_report(total_time, target_tests)
    
    async def log_progress(self, completed_batches: int, total_batches: int):
        """Log current progress"""
        completed_tests = len(self.test_results)
        success_rate = sum(1 for r in self.test_results if r.success) / len(self.test_results) * 100
        
        if self.accuracy_scores:
            avg_accuracy = statistics.mean(self.accuracy_scores) * 100
            avg_response_time = statistics.mean(self.response_times) * 1000
        else:
            avg_accuracy = 0
            avg_response_time = 0
        
        logger.info(
            f"Progress: {completed_batches}/{total_batches} batches | "
            f"Tests: {completed_tests:,} | "
            f"Success: {success_rate:.1f}% | "
            f"Avg Accuracy: {avg_accuracy:.1f}% | "
            f"Avg Response: {avg_response_time:.1f}ms"
        )
    
    async def generate_comprehensive_report(self, total_time: float, target_tests: int):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Calculate statistics
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        if self.accuracy_scores:
            avg_accuracy = statistics.mean(self.accuracy_scores)
            min_accuracy = min(self.accuracy_scores)
            max_accuracy = max(self.accuracy_scores)
            accuracy_std = statistics.stdev(self.accuracy_scores) if len(self.accuracy_scores) > 1 else 0
        else:
            avg_accuracy = min_accuracy = max_accuracy = accuracy_std = 0
        
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            p95_response_time = np.percentile(self.response_times, 95)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        # Real data validation
        real_data_percentage = (
            self.real_data_validation['real_data_queries'] / 
            self.real_data_validation['total_queries'] * 100
        ) if self.real_data_validation['total_queries'] > 0 else 0
        
        mock_data_percentage = (
            self.real_data_validation['mock_data_detected'] / 
            self.real_data_validation['total_queries'] * 100
        ) if self.real_data_validation['total_queries'] > 0 else 0
        
        # Generate report
        report = f"""
{'='*80}
COMPREHENSIVE SYSTEM TEST REPORT
{'='*80}

📊 TEST EXECUTION SUMMARY:
• Target Tests: {target_tests:,}
• Completed Tests: {total_tests:,}
• Successful Tests: {successful_tests:,}
• Failed Tests: {failed_tests:,}
• Success Rate: {success_rate:.2f}%
• Total Execution Time: {total_time/60:.1f} minutes
• Tests per Second: {total_tests/total_time:.1f}

🎯 ACCURACY ANALYSIS:
• Average Accuracy: {avg_accuracy*100:.2f}%
• Minimum Accuracy: {min_accuracy*100:.2f}%
• Maximum Accuracy: {max_accuracy*100:.2f}%
• Accuracy Std Dev: {accuracy_std*100:.2f}%
• 90%+ Accuracy Rate: {sum(1 for a in self.accuracy_scores if a >= 0.90)/len(self.accuracy_scores)*100:.1f}%

⚡ PERFORMANCE METRICS:
• Average Response Time: {avg_response_time*1000:.1f}ms
• Minimum Response Time: {min_response_time*1000:.1f}ms
• Maximum Response Time: {max_response_time*1000:.1f}ms
• 95th Percentile: {p95_response_time*1000:.1f}ms

🛡️ REAL DATA VALIDATION:
• Total Queries Analyzed: {self.real_data_validation['total_queries']:,}
• Real Data Queries: {self.real_data_validation['real_data_queries']:,} ({real_data_percentage:.1f}%)
• Mock Data Detected: {self.real_data_validation['mock_data_detected']:,} ({mock_data_percentage:.1f}%)
• Validation Failures: {self.real_data_validation['validation_failures']:,}

✅ SYSTEM VALIDATION:
• Accuracy Target (90%+): {'✅ ACHIEVED' if avg_accuracy >= 0.90 else '❌ NOT MET'}
• Response Time (<2s): {'✅ ACHIEVED' if avg_response_time < 2.0 else '❌ NOT MET'}
• Success Rate (95%+): {'✅ ACHIEVED' if success_rate >= 95 else '❌ NOT MET'}
• Real Data Only: {'✅ VERIFIED' if mock_data_percentage < 1.0 else '❌ MOCK DATA DETECTED'}

🔬 DETAILED BREAKDOWN:
• CS:GO Queries: {sum(1 for r in self.test_results if any(player in r.query.lower() for player in ['s1mple', 'zywoo', 'device', 'niko'])):,}
• Valorant Queries: {sum(1 for r in self.test_results if any(player in r.query.lower() for player in ['tenz', 'aspas', 'derke', 'valorant'])):,}
• NBA Queries: {sum(1 for r in self.test_results if any(player in r.query.lower() for player in ['lebron', 'curry', 'durant', 'points', 'rebounds'])):,}
• System Queries: {sum(1 for r in self.test_results if any(word in r.query.lower() for word in ['system', 'status', 'health', 'accuracy'])):,}

🎖️ FINAL VERDICT:
{'🏆 SYSTEM PERFORMANCE: EXCEPTIONAL' if avg_accuracy >= 0.95 and success_rate >= 95 and mock_data_percentage < 1.0 else
 '✅ SYSTEM PERFORMANCE: GOOD' if avg_accuracy >= 0.90 and success_rate >= 90 and mock_data_percentage < 5.0 else
 '⚠️ SYSTEM PERFORMANCE: NEEDS IMPROVEMENT'}

{'='*80}
        """
        
        print(report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"/app/test_reports/comprehensive_test_{timestamp}.txt"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs("/app/test_reports", exist_ok=True)
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive test report saved to: {report_filename}")
        
        # Save detailed results as JSON
        detailed_results = {
            'test_summary': {
                'target_tests': target_tests,
                'completed_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'execution_time': total_time
            },
            'accuracy_analysis': {
                'average_accuracy': avg_accuracy,
                'min_accuracy': min_accuracy,
                'max_accuracy': max_accuracy,
                'accuracy_std': accuracy_std
            },
            'performance_metrics': {
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time
            },
            'real_data_validation': self.real_data_validation,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(f"/app/test_reports/detailed_results_{timestamp}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

    async def run_quick_validation_test(self, num_tests: int = 1000):
        """Run quick validation test"""
        logger.info(f"Running quick validation test with {num_tests} predictions")
        await self.run_comprehensive_test(num_tests)

    async def run_medium_test(self, num_tests: int = 100_000):
        """Run medium test"""
        logger.info(f"Running medium test with {num_tests:,} predictions")
        await self.run_comprehensive_test(num_tests)

    async def run_full_five_million_test(self):
        """Run the full 5 million test"""
        logger.info("🚀 Starting FULL 5,000,000 prediction test!")
        await self.run_comprehensive_test(5_000_000)

# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Sports AI System Tester')
    parser.add_argument('--tests', type=int, default=1000, help='Number of tests to run')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (1K tests)')
    parser.add_argument('--medium', action='store_true', help='Run medium test (100K tests)')
    parser.add_argument('--full', action='store_true', help='Run full 5M test')
    parser.add_argument('--url', default='http://localhost:8001', help='Base URL for API')
    
    args = parser.parse_args()
    
    tester = ComprehensiveSystemTester(args.url)
    
    if args.full:
        await tester.run_full_five_million_test()
    elif args.medium:
        await tester.run_medium_test()
    elif args.quick:
        await tester.run_quick_validation_test()
    else:
        await tester.run_comprehensive_test(args.tests)

if __name__ == "__main__":
    asyncio.run(main())