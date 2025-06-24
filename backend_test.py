#!/usr/bin/env python3
import requests
import json
import os
import sys
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

    def test_players_endpoint(self):
        """Test the players endpoint"""
        try:
            response = requests.get(f"{self.api_url}/players")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response data: {json.dumps(data, indent=2)}")
            
            if not isinstance(data, dict) or "players" not in data:
                print("Error: Response does not contain 'players' key")
                return False
            
            players = data["players"]
            if not isinstance(players, list) or len(players) == 0:
                print("Error: 'players' is not a list or is empty")
                return False
            
            expected_players = ["lebron james", "stephen curry", "patrick mahomes"]
            for player in expected_players:
                if player not in players:
                    print(f"Error: Expected player '{player}' not found in response")
                    return False
            
            print(f"Found {len(players)} players in the response")
            return True
        except Exception as e:
            print(f"Error in players endpoint test: {e}")
            return False

    def test_teams_endpoint(self):
        """Test the teams endpoint"""
        try:
            response = requests.get(f"{self.api_url}/teams")
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error: Unexpected status code {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response data: {json.dumps(data, indent=2)}")
            
            if not isinstance(data, dict) or "teams" not in data:
                print("Error: Response does not contain 'teams' key")
                return False
            
            teams = data["teams"]
            if not isinstance(teams, list) or len(teams) == 0:
                print("Error: 'teams' is not a list or is empty")
                return False
            
            expected_teams = ["lakers", "warriors", "chiefs"]
            for team in expected_teams:
                if team not in teams:
                    print(f"Error: Expected team '{team}' not found in response")
                    return False
            
            print(f"Found {len(teams)} teams in the response")
            return True
        except Exception as e:
            print(f"Error in teams endpoint test: {e}")
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

    def test_chat_endpoint_over_under(self):
        """Test the chat endpoint with an over/under question"""
        try:
            message = "Will LeBron James score over 22 points?"
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
                "LeBron James", 
                "Analysis", 
                "Line:", 
                "Season Average:", 
                "Recent 3-Game Average:", 
                "Odds:", 
                "Recommendation:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint over/under test: {e}")
            return False

    def test_chat_endpoint_player_stats(self):
        """Test the chat endpoint with a player stats question"""
        try:
            message = "Stephen Curry stats"
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
                "Stephen Curry", 
                "Season Stats", 
                "Team:", 
                "Position:", 
                "Season Averages:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint player stats test: {e}")
            return False

    def test_chat_endpoint_team_info(self):
        """Test the chat endpoint with a team info question"""
        try:
            message = "Lakers record"
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
                "Los Angeles Lakers", 
                "Team Info", 
                "Record:", 
                "Next Game:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint team info test: {e}")
            return False

    def test_chat_endpoint_fanduel_lineup(self):
        """Test the chat endpoint with a FanDuel lineup question"""
        try:
            message = "FanDuel lineup suggestions"
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
            
            # Check for expected content in the response - note that the API returns "Fanduel" (lowercase 'd')
            expected_content = [
                "Fanduel", 
                "Lineup Suggestion", 
                "Salary Cap:", 
                "Recommended Players:", 
                "Total Used:", 
                "Total Projection:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint FanDuel lineup test: {e}")
            return False

    def test_chat_endpoint_parlay(self):
        """Test the chat endpoint with a parlay question"""
        try:
            message = "Build me a parlay"
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
                "Parlay Builder", 
                "Same Game Parlay", 
                "Multi-Game Parlay", 
                "Tips:"
            ]
            
            for content in expected_content:
                if content not in response_text:
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
        except Exception as e:
            print(f"Error in chat endpoint parlay test: {e}")
            return False

    def test_chat_endpoint_unknown_player(self):
        """Test the chat endpoint with an unknown player"""
        try:
            message = "Will Michael Jordan score over 30 points?"
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
            
            # Check for expected content in the response - note that the API returns lowercase player name
            expected_content = [
                "Sorry", 
                "don't have data", 
                "michael jordan"
            ]
            
            for content in expected_content:
                if content not in response_text.lower():
                    print(f"Error: Expected content '{content}' not found in response")
                    return False
            
            return True
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
                "Betting Lines", 
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
        print(f"\n{'='*80}\nRunning Sports Agent Backend Tests\n{'='*80}")
        
        # Run all tests
        self.run_test("Health Check", self.test_health_check)
        self.run_test("Players Endpoint", self.test_players_endpoint)
        self.run_test("Teams Endpoint", self.test_teams_endpoint)
        self.run_test("Chat History Endpoint", self.test_chat_history_endpoint)
        self.run_test("Chat Endpoint - Over/Under", self.test_chat_endpoint_over_under)
        self.run_test("Chat Endpoint - Player Stats", self.test_chat_endpoint_player_stats)
        self.run_test("Chat Endpoint - Team Info", self.test_chat_endpoint_team_info)
        self.run_test("Chat Endpoint - FanDuel Lineup", self.test_chat_endpoint_fanduel_lineup)
        self.run_test("Chat Endpoint - Parlay", self.test_chat_endpoint_parlay)
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