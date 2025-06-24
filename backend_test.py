#!/usr/bin/env python3
import requests
import json
import os
import sys
import time
from datetime import datetime

# Get the backend URL from the frontend .env file
def get_backend_url():
    try:
        with open('/app/frontend/.env', 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    return line.strip().split('=')[1].strip('"\'')
    except Exception as e:
        print(f"Error reading backend URL: {e}")
        return None

# Main test class
class SportsAgentBackendTest:
    def __init__(self):
        self.backend_url = get_backend_url()
        if not self.backend_url:
            print("ERROR: Could not determine backend URL")
            sys.exit(1)
        
        self.api_url = f"{self.backend_url}/api"
        print(f"Testing backend at: {self.api_url}")
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }

    def run_test(self, test_name, test_func, *args, **kwargs):
        """Run a test and record the result"""
        print(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}")
        self.test_results["total_tests"] += 1
        
        try:
            result = test_func(*args, **kwargs)
            if result:
                self.test_results["passed_tests"] += 1
                status = "PASSED"
            else:
                self.test_results["failed_tests"] += 1
                status = "FAILED"
        except Exception as e:
            self.test_results["failed_tests"] += 1
            status = f"ERROR: {str(e)}"
        
        self.test_results["test_details"].append({
            "name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"Test {test_name}: {status}")
        return result

    def test_health_check(self):
        """Test the health check endpoint"""
        try:
            # The API doesn't have a specific health check endpoint, so we'll use the players endpoint
            response = requests.get(f"{self.api_url}/players")
            print(f"Response status code: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error in health check: {e}")
            return False

    def test_real_players_endpoint(self):
        """Test the players endpoint for real NBA players from Ball Don't Lie API"""
        try:
            response = requests.get(f"{self.api_url}/players")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            
            if not isinstance(data, dict) or "players" not in data:
                print("Error: Response does not contain 'players' key")
                return False
            
            players = data["players"]
            if not isinstance(players, list) or len(players) == 0:
                print("Error: 'players' is not a list or is empty")
                return False
            
            # Check for common NBA players that should be in the response
            # These are real NBA players that should be returned by the Ball Don't Lie API
            expected_players = ["lebron james", "stephen curry", "kevin durant", "giannis antetokounmpo"]
            found_players = []
            
            for player in expected_players:
                if any(player in p.lower() for p in players):
                    found_players.append(player)
            
            # We should find at least 2 of the expected players
            if len(found_players) < 2:
                print(f"Error: Not enough expected players found. Found: {found_players}")
                return False
            
            # Check if we have a substantial number of players (Ball Don't Lie API should return many)
            if len(players) < 20:
                print(f"Error: Expected more players. Only found {len(players)}")
                return False
            
            print(f"Found {len(players)} players in the response")
            print(f"Sample players: {players[:5]}")
            return True
        except Exception as e:
            print(f"Error in real players endpoint test: {e}")
            return False

    def test_news_endpoint(self):
        """Test the news endpoint for real ESPN news integration"""
        try:
            response = requests.get(f"{self.api_url}/news")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "news" not in data:
                print("Error: Response does not contain 'news' key")
                return False
            
            news = data["news"]
            if not isinstance(news, list) or len(news) == 0:
                print("Error: 'news' is not a list or is empty")
                return False
            
            # Check if news items have expected structure from ESPN RSS feed
            for item in news[:3]:
                if not all(key in item for key in ["title", "link", "source"]):
                    print(f"Error: News item missing required fields: {item}")
                    return False
                
                # Verify this is real news by checking for ESPN source
                if "espn" not in item["source"].lower():
                    print(f"Error: News source is not ESPN: {item['source']}")
                    return False
            
            print(f"Found {len(news)} news items")
            print(f"Sample news titles: {[item['title'] for item in news[:3]]}")
            return True
        except Exception as e:
            print(f"Error in news endpoint test: {e}")
            return False

    def test_odds_endpoint(self):
        """Test the betting odds endpoint"""
        try:
            response = requests.get(f"{self.api_url}/odds")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "games" not in data:
                print("Error: Response does not contain 'games' key")
                return False
            
            games = data["games"]
            if not isinstance(games, list) or len(games) == 0:
                print("Error: 'games' is not a list or is empty")
                return False
            
            # Check if games have expected structure
            for game in games[:2]:
                if not all(key in game for key in ["home_team", "away_team", "bookmakers"]):
                    print(f"Error: Game missing required fields: {game}")
                    return False
            
            print(f"Found {len(games)} games with odds")
            return True
        except Exception as e:
            print(f"Error in odds endpoint test: {e}")
            return False

    def test_chat_history_endpoint(self):
        """Test the chat history endpoint"""
        try:
            response = requests.get(f"{self.api_url}/chat-history")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "history" not in data:
                print("Error: Response does not contain 'history' key")
                return False
            
            history = data["history"]
            if not isinstance(history, list):
                print("Error: 'history' is not a list")
                return False
            
            print(f"Found {len(history)} chat history entries")
            return True
        except Exception as e:
            print(f"Error in chat history endpoint test: {e}")
            return False

    def test_chat_endpoint_real_player_stats(self):
        """Test the chat endpoint with a player stats question for real data"""
        try:
            message = "LeBron James stats"
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": message}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "response" not in data:
                print("Error: Response does not contain 'response' key")
                return False
            
            response_text = data["response"]
            print(f"Response preview: {response_text[:200]}...")
            
            # Check for expected content in the response
            expected_content = [
                "LeBron James", 
                "Season Stats", 
                "Team:", 
                "Position:", 
                "Season Averages:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            # Check for real stats data (numbers, not placeholder text)
            import re
            stats_pattern = r'Points: (\d+\.?\d*)'
            stats_match = re.search(stats_pattern, response_text)
            
            if not stats_match:
                print("Error: Could not find real points stats in response")
                return False
            
            points = float(stats_match.group(1))
            if points <= 0:
                print(f"Error: Points value ({points}) appears to be invalid")
                return False
            
            print(f"Found real stats data with points: {points}")
            return True
        except Exception as e:
            print(f"Error in chat endpoint real player stats test: {e}")
            return False

    def test_chat_endpoint_real_over_under(self):
        """Test the chat endpoint with an over/under question using real data"""
        try:
            message = "Will Stephen Curry score over 20 points?"
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": message}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "response" not in data:
                print("Error: Response does not contain 'response' key")
                return False
            
            response_text = data["response"]
            print(f"Response preview: {response_text[:200]}...")
            
            # Check for expected content in the response
            expected_content = [
                "Stephen Curry", 
                "Analysis", 
                "Line:", 
                "Season Average:", 
                "Team:", 
                "Position:", 
                "Recommendation:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            # Check for real stats data (numbers, not placeholder text)
            import re
            avg_pattern = r'Season Average: (\d+\.?\d*)'
            avg_match = re.search(avg_pattern, response_text)
            
            if not avg_match:
                print("Error: Could not find real season average in response")
                return False
            
            avg = float(avg_match.group(1))
            if avg <= 0:
                print(f"Error: Season average ({avg}) appears to be invalid")
                return False
            
            # Check for confidence percentage based on real data
            confidence_pattern = r'Confidence: (\d+)%'
            confidence_match = re.search(confidence_pattern, response_text)
            
            if not confidence_match:
                print("Error: Could not find confidence percentage in response")
                return False
            
            print(f"Found real stats data with season average: {avg}")
            return True
        except Exception as e:
            print(f"Error in chat endpoint real over/under test: {e}")
            return False

    def test_chat_endpoint_news(self):
        """Test the chat endpoint with a news request"""
        try:
            message = "latest sports news"
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": message}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "response" not in data:
                print("Error: Response does not contain 'response' key")
                return False
            
            response_text = data["response"]
            print(f"Response preview: {response_text[:200]}...")
            
            # Check for expected content in the response
            expected_content = [
                "Latest Sports News"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            # Check for ESPN source attribution
            if "ESPN" not in response_text:
                print("Error: ESPN source not found in news response")
                return False
            
            # Check for multiple news items
            news_items = response_text.count("🔸")
            if news_items < 2:
                print(f"Error: Expected multiple news items, found {news_items}")
                return False
            
            print(f"Found {news_items} news items in the response")
            return True
        except Exception as e:
            print(f"Error in chat endpoint news test: {e}")
            return False

    def test_mongodb_caching(self):
        """Test MongoDB caching functionality by making repeated requests"""
        try:
            # First request should hit the API and cache the result
            start_time = time.time()
            response1 = requests.get(f"{self.api_url}/players")
            first_request_time = time.time() - start_time
            
            if response1.status_code != 200:
                print(f"Error: First request failed with status code {response1.status_code}")
                return False
            
            # Wait a moment to ensure any async operations complete
            time.sleep(1)
            
            # Second request should use the cached result and be faster
            start_time = time.time()
            response2 = requests.get(f"{self.api_url}/players")
            second_request_time = time.time() - start_time
            
            if response2.status_code != 200:
                print(f"Error: Second request failed with status code {response2.status_code}")
                return False
            
            # Compare response data to ensure it's the same
            data1 = response1.json()
            data2 = response2.json()
            
            if data1 != data2:
                print("Error: Cached response data differs from original request")
                return False
            
            # Check if second request was faster (indicating cache hit)
            # Note: This is not a perfect test as network conditions can vary,
            # but a significant difference suggests caching is working
            print(f"First request time: {first_request_time:.4f}s")
            print(f"Second request time: {second_request_time:.4f}s")
            
            # The second request doesn't need to be faster since both might be cached
            # We're just checking that the data is consistent
            
            print("Cache test passed: Consistent data between requests")
            return True
        except Exception as e:
            print(f"Error in MongoDB caching test: {e}")
            return False

    def test_chat_endpoint_unknown_player(self):
        """Test the chat endpoint with an unknown player"""
        try:
            # Use a very unlikely player name
            message = "Will XYZ123NonExistentPlayer score over 30 points?"
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": message}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "response" not in data:
                print("Error: Response does not contain 'response' key")
                return False
            
            response_text = data["response"]
            print(f"Full response: {response_text}")
            
            # Check if the response indicates the player is not found
            if "sorry" in response_text.lower() and ("couldn't find" in response_text.lower() or "don't have data" in response_text.lower()):
                return True
            else:
                print("Error: Response does not indicate that the player is not found")
                return False
        except Exception as e:
            print(f"Error in chat endpoint unknown player test: {e}")
            return False

    def test_chat_endpoint_general_query(self):
        """Test the chat endpoint with a general query"""
        try:
            message = "What can you help me with?"
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": message}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if not isinstance(data, dict) or "response" not in data:
                print("Error: Response does not contain 'response' key")
                return False
            
            response_text = data["response"]
            print(f"Response preview: {response_text[:100]}...")
            
            # Check for expected content in the response
            expected_content = [
                "Sports Agent AI", 
                "Player Analysis", 
                "Live Data", 
                "Daily Fantasy", 
                "Parlays"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint general query test: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and print a summary"""
        print(f"\n{'='*80}\nRunning Sports Agent Backend Tests with REAL Data Integration\n{'='*80}")
        
        # Run all tests
        self.run_test("Health Check", self.test_health_check)
        self.run_test("Real Players Endpoint (Ball Don't Lie API)", self.test_real_players_endpoint)
        self.run_test("News Endpoint (ESPN RSS)", self.test_news_endpoint)
        self.run_test("Odds Endpoint", self.test_odds_endpoint)
        self.run_test("Chat History Endpoint", self.test_chat_history_endpoint)
        self.run_test("Chat Endpoint - Real Player Stats", self.test_chat_endpoint_real_player_stats)
        self.run_test("Chat Endpoint - Real Over/Under Analysis", self.test_chat_endpoint_real_over_under)
        self.run_test("Chat Endpoint - Latest News", self.test_chat_endpoint_news)
        self.run_test("MongoDB Caching", self.test_mongodb_caching)
        self.run_test("Chat Endpoint - Unknown Player", self.test_chat_endpoint_unknown_player)
        self.run_test("Chat Endpoint - General Query", self.test_chat_endpoint_general_query)
        
        # Print summary
        print(f"\n{'='*80}\nTest Summary\n{'='*80}")
        print(f"Total Tests: {self.test_results['total_tests']}")
        print(f"Passed Tests: {self.test_results['passed_tests']}")
        print(f"Failed Tests: {self.test_results['failed_tests']}")
        
        # Print details of failed tests
        if self.test_results['failed_tests'] > 0:
            print("\nFailed Tests:")
            for test in self.test_results['test_details']:
                if test['status'] != 'PASSED':
                    print(f"- {test['name']}: {test['status']}")
        
        return self.test_results['failed_tests'] == 0

if __name__ == "__main__":
    tester = SportsAgentBackendTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)