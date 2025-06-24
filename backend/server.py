from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import re
import random
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Mock Sports Data
PLAYERS = {
    "lebron james": {
        "name": "LeBron James",
        "team": "Los Angeles Lakers",
        "position": "Forward",
        "season_avg": {
            "points": 25.3,
            "rebounds": 7.3,
            "assists": 7.4,
            "minutes": 35.5
        },
        "recent_games": [
            {"date": "2025-06-20", "points": 28, "rebounds": 6, "assists": 11},
            {"date": "2025-06-18", "points": 19, "rebounds": 8, "assists": 6},
            {"date": "2025-06-15", "points": 30, "rebounds": 9, "assists": 9}
        ]
    },
    "stephen curry": {
        "name": "Stephen Curry",
        "team": "Golden State Warriors",
        "position": "Guard",
        "season_avg": {
            "points": 29.5,
            "rebounds": 4.9,
            "assists": 6.2,
            "threes": 4.8
        },
        "recent_games": [
            {"date": "2025-06-20", "points": 32, "rebounds": 4, "assists": 7, "threes": 6},
            {"date": "2025-06-18", "points": 27, "rebounds": 5, "assists": 8, "threes": 5},
            {"date": "2025-06-15", "points": 35, "rebounds": 3, "assists": 4, "threes": 7}
        ]
    },
    "patrick mahomes": {
        "name": "Patrick Mahomes",
        "team": "Kansas City Chiefs",
        "position": "Quarterback",
        "season_avg": {
            "passing_yards": 287.2,
            "touchdowns": 2.1,
            "completions": 24.8,
            "rating": 98.5
        },
        "recent_games": [
            {"date": "2025-06-15", "passing_yards": 312, "touchdowns": 3, "completions": 28},
            {"date": "2025-06-08", "passing_yards": 268, "touchdowns": 2, "completions": 22},
            {"date": "2025-06-01", "passing_yards": 295, "touchdowns": 1, "completions": 26}
        ]
    }
}

TEAMS = {
    "lakers": {"name": "Los Angeles Lakers", "record": "45-20", "next_game": "vs Warriors"},
    "warriors": {"name": "Golden State Warriors", "record": "42-23", "next_game": "@ Lakers"},
    "chiefs": {"name": "Kansas City Chiefs", "record": "12-2", "next_game": "vs Bills"}
}

# Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str

class BettingOdds(BaseModel):
    player: str
    stat: str
    line: float
    over_odds: int
    under_odds: int
    confidence: int

class SportsQuery:
    def __init__(self):
        self.patterns = {
            'over_under': r'(will|can) (.+?) (score|get|have) (over|under) (\d+\.?\d*) (points|rebounds|assists|yards|touchdowns)',
            'spread': r'(.+?) (spread|line) (.+?) vs (.+)',
            'moneyline': r'(.+?) (win|beat) (.+?) moneyline',
            'parlay': r'parlay (.+?) and (.+)',
            'lineup': r'(fanduel|draftkings) lineup (.+)',
            'player_stats': r'(.+?) (stats|average|season)',
            'team_info': r'(.+?) (record|next game|info)'
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        query = query.lower().strip()
        
        # Over/Under Pattern
        over_under_match = re.search(self.patterns['over_under'], query)
        if over_under_match:
            player_name = over_under_match.group(2).strip()
            stat_type = over_under_match.group(6)
            line_value = float(over_under_match.group(5))
            over_under = over_under_match.group(4)
            
            return {
                'type': 'over_under',
                'player': player_name,
                'stat': stat_type,
                'line': line_value,
                'direction': over_under
            }
        
        # Player Stats Pattern
        stats_match = re.search(self.patterns['player_stats'], query)
        if stats_match:
            player_name = stats_match.group(1).strip()
            return {
                'type': 'player_stats',
                'player': player_name
            }
        
        # Team Info Pattern
        team_match = re.search(self.patterns['team_info'], query)
        if team_match:
            team_name = team_match.group(1).strip()
            return {
                'type': 'team_info',
                'team': team_name
            }
        
        return {'type': 'general', 'query': query}

    def generate_betting_odds(self, player: str, stat: str, line: float) -> BettingOdds:
        # Generate realistic odds
        over_odds = random.choice([-110, -105, -115, -120, +100, +105])
        under_odds = random.choice([-110, -105, -115, -120, +100, +105])
        confidence = random.randint(65, 90)
        
        return BettingOdds(
            player=player,
            stat=stat,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            confidence=confidence
        )

    def get_player_analysis(self, player_name: str, stat: str, line: float, direction: str) -> str:
        player_key = player_name.lower()
        if player_key not in PLAYERS:
            return f"Sorry, I don't have data for {player_name}. Try players like LeBron James, Stephen Curry, or Patrick Mahomes."
        
        player_data = PLAYERS[player_key]
        season_avg = player_data['season_avg'].get(stat.rstrip('s'), 0)
        recent_games = player_data['recent_games']
        
        # Calculate recent average
        recent_stat_values = []
        for game in recent_games:
            if stat.rstrip('s') in game:
                recent_stat_values.append(game[stat.rstrip('s')])
        
        recent_avg = sum(recent_stat_values) / len(recent_stat_values) if recent_stat_values else season_avg
        
        odds = self.generate_betting_odds(player_data['name'], stat, line)
        
        analysis = f"📊 **{player_data['name']} - {stat.title()} Analysis**\n\n"
        analysis += f"🎯 **Line:** {direction.title()} {line} {stat}\n"
        analysis += f"📈 **Season Average:** {season_avg}\n"
        analysis += f"🔥 **Recent 3-Game Average:** {recent_avg:.1f}\n\n"
        
        if direction == "over":
            hit_rate = "68%" if recent_avg > line else "45%"
            recommendation = "✅ LEAN OVER" if recent_avg > line else "❌ LEAN UNDER"
        else:
            hit_rate = "72%" if recent_avg < line else "38%"
            recommendation = "✅ LEAN UNDER" if recent_avg < line else "❌ LEAN OVER"
        
        analysis += f"🎲 **Odds:** Over {odds.over_odds:+d} | Under {odds.under_odds:+d}\n"
        analysis += f"📊 **Hit Rate:** {hit_rate}\n"
        analysis += f"🎯 **Recommendation:** {recommendation}\n"
        analysis += f"🔒 **Confidence:** {odds.confidence}%\n\n"
        
        analysis += f"**Recent Games:**\n"
        for i, game in enumerate(recent_games[:3], 1):
            stat_value = game.get(stat.rstrip('s'), 'N/A')
            hit_miss = "✅" if stat_value != 'N/A' and ((direction == "over" and stat_value > line) or (direction == "under" and stat_value < line)) else "❌"
            analysis += f"{i}. {game['date']}: {stat_value} {stat.rstrip('s')} {hit_miss}\n"
        
        return analysis

    def get_player_stats(self, player_name: str) -> str:
        player_key = player_name.lower()
        if player_key not in PLAYERS:
            return f"Sorry, I don't have data for {player_name}. Try players like LeBron James, Stephen Curry, or Patrick Mahomes."
        
        player_data = PLAYERS[player_key]
        stats = player_data['season_avg']
        
        response = f"📊 **{player_data['name']} - Season Stats**\n\n"
        response += f"🏀 **Team:** {player_data['team']}\n"
        response += f"📍 **Position:** {player_data['position']}\n\n"
        response += "**Season Averages:**\n"
        
        for stat, value in stats.items():
            stat_name = stat.replace('_', ' ').title()
            response += f"• {stat_name}: {value}\n"
        
        return response

    def get_team_info(self, team_name: str) -> str:
        team_key = team_name.lower()
        if team_key not in TEAMS:
            return f"Sorry, I don't have data for {team_name}. Try teams like Lakers, Warriors, or Chiefs."
        
        team_data = TEAMS[team_key]
        
        response = f"🏆 **{team_data['name']} - Team Info**\n\n"
        response += f"📊 **Record:** {team_data['record']}\n"
        response += f"🎮 **Next Game:** {team_data['next_game']}\n"
        
        return response

    def generate_lineup_suggestion(self, platform: str) -> str:
        lineups = {
            'fanduel': {
                'salary_cap': 60000,
                'positions': ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C'],
                'players': [
                    {'name': 'Stephen Curry', 'position': 'PG', 'salary': 11500, 'projection': 52.3},
                    {'name': 'LeBron James', 'position': 'SF', 'salary': 10800, 'projection': 48.7},
                    {'name': 'Giannis Antetokounmpo', 'position': 'PF', 'salary': 11200, 'projection': 55.1},
                ]
            },
            'draftkings': {
                'salary_cap': 50000,
                'positions': ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'],
                'players': [
                    {'name': 'Luka Doncic', 'position': 'PG', 'salary': 11000, 'projection': 58.2},
                    {'name': 'Jayson Tatum', 'position': 'SF', 'salary': 9800, 'projection': 46.5},
                    {'name': 'Joel Embiid', 'position': 'C', 'salary': 10500, 'projection': 52.8},
                ]
            }
        }
        
        platform_data = lineups.get(platform.lower(), lineups['fanduel'])
        
        response = f"🎯 **{platform.title()} Lineup Suggestion**\n\n"
        response += f"💰 **Salary Cap:** ${platform_data['salary_cap']:,}\n\n"
        response += "**Recommended Players:**\n"
        
        total_salary = 0
        total_projection = 0
        
        for player in platform_data['players']:
            response += f"• {player['name']} ({player['position']}) - ${player['salary']:,} | Proj: {player['projection']}\n"
            total_salary += player['salary']
            total_projection += player['projection']
        
        response += f"\n💵 **Total Used:** ${total_salary:,}\n"
        response += f"📈 **Total Projection:** {total_projection:.1f} points\n"
        response += f"💡 **Value Rating:** ⭐⭐⭐⭐"
        
        return response

sports_query = SportsQuery()

@api_router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        query = request.message.strip()
        parsed_query = sports_query.parse_query(query)
        
        if parsed_query['type'] == 'over_under':
            response = sports_query.get_player_analysis(
                parsed_query['player'],
                parsed_query['stat'],
                parsed_query['line'],
                parsed_query['direction']
            )
        elif parsed_query['type'] == 'player_stats':
            response = sports_query.get_player_stats(parsed_query['player'])
        elif parsed_query['type'] == 'team_info':
            response = sports_query.get_team_info(parsed_query['team'])
        elif 'fanduel' in query.lower() or 'draftkings' in query.lower():
            platform = 'fanduel' if 'fanduel' in query.lower() else 'draftkings'
            response = sports_query.generate_lineup_suggestion(platform)
        elif 'parlay' in query.lower():
            response = "🎲 **Parlay Builder**\n\nGreat choice! Parlays can offer big payouts. Here are some popular parlay combinations:\n\n🏀 **Same Game Parlay Ideas:**\n• LeBron Over 25.5 Points + Lakers ML\n• Curry Over 4.5 Threes + Warriors +3.5\n\n🏈 **Multi-Game Parlay:**\n• Chiefs ML + Over 47.5 Total Points\n• Mahomes Over 275.5 Passing Yards + 2+ TDs\n\n💡 **Tips:**\n• Start with 2-3 legs for better odds\n• Avoid correlated bets\n• Bet responsibly!"
        else:
            response = f"🏈🏀 **Sports Agent AI** 🏀🏈\n\nI can help you with:\n\n📊 **Player Analysis:**\n• \"Will LeBron James score over 22 points?\"\n• \"Stephen Curry stats\"\n\n🎲 **Betting Lines:**\n• Over/Under predictions\n• Spread analysis\n• Moneyline picks\n\n🎯 **Daily Fantasy:**\n• \"FanDuel lineup suggestions\"\n• \"DraftKings optimal picks\"\n\n🎪 **Parlays:**\n• \"Build me a parlay\"\n• Same-game parlays\n\nTry asking about your favorite players!"
        
        # Save chat to database
        chat_message = ChatMessage(message=query, response=response)
        await db.chat_history.insert_one(chat_message.dict())
        
        return {"response": response}
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

@api_router.get("/players")
async def get_players():
    return {"players": list(PLAYERS.keys())}

@api_router.get("/teams")
async def get_teams():
    return {"teams": list(TEAMS.keys())}

@api_router.get("/chat-history")
async def get_chat_history():
    try:
        history = await db.chat_history.find().sort("timestamp", -1).limit(50).to_list(50)
        return {"history": [ChatMessage(**chat) for chat in history]}
    except Exception as e:
        logging.error(f"Error getting chat history: {str(e)}")
        return {"history": []}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()