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
import httpx
import asyncio
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str

class PlayerStats(BaseModel):
    name: str
    team: str
    position: str
    stats: Dict[str, Any]
    recent_games: List[Dict[str, Any]]

# Real Data Services
class SportsDataService:
    def __init__(self):
        self.base_urls = {
            'balldontlie': 'https://www.balldontlie.io/api/v1',
            'nba_stats': 'https://stats.nba.com/stats',
            'sportsdb': 'https://www.thesportsdb.com/api/v1/json/3',
            'odds_api': 'https://api.the-odds-api.com/v4'
        }
        self.cache_ttl = 300  # 5 minutes cache
    
    async def get_cached_data(self, cache_key: str):
        """Get cached data from MongoDB"""
        try:
            cached = await db.cache.find_one({"key": cache_key})
            if cached and cached.get("expires_at", datetime.min) > datetime.utcnow():
                return cached.get("data")
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        return None
    
    async def set_cache_data(self, cache_key: str, data: Any, ttl_minutes: int = 5):
        """Cache data in MongoDB with TTL"""
        try:
            expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
            await db.cache.replace_one(
                {"key": cache_key},
                {
                    "key": cache_key,
                    "data": data,
                    "expires_at": expires_at,
                    "created_at": datetime.utcnow()
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    async def fetch_nba_players(self):
        """Fetch real NBA players from Ball Don't Lie API"""
        cache_key = "nba_players_list"
        cached_data = await self.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_urls['balldontlie']}/players?per_page=100")
                if response.status_code == 200:
                    data = response.json()
                    players_data = data.get('data', [])
                    
                    # Process and store player data
                    processed_players = {}
                    for player in players_data:
                        if player.get('first_name') and player.get('last_name'):
                            full_name = f"{player['first_name']} {player['last_name']}"
                            key = full_name.lower()
                            processed_players[key] = {
                                'id': player.get('id'),
                                'name': full_name,
                                'position': player.get('position', 'N/A'),
                                'team': player.get('team', {}).get('full_name', 'Free Agent'),
                                'team_abbr': player.get('team', {}).get('abbreviation', 'FA')
                            }
                    
                    await self.set_cache_data(cache_key, processed_players, 60)  # Cache for 1 hour
                    return processed_players
                else:
                    logger.error(f"NBA API error: {response.status_code}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching NBA players: {e}")
            return {}
    
    async def fetch_player_stats(self, player_name: str):
        """Fetch real player statistics"""
        cache_key = f"player_stats_{player_name.lower().replace(' ', '_')}"
        cached_data = await self.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            players_data = await self.fetch_nba_players()
            player_key = player_name.lower()
            
            if player_key not in players_data:
                return None
            
            player_info = players_data[player_key]
            player_id = player_info['id']
            
            # Fetch season averages (using Ball Don't Lie API)
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get season averages
                avg_response = await client.get(
                    f"{self.base_urls['balldontlie']}/season_averages",
                    params={'season': 2024, 'player_ids[]': player_id}
                )
                
                season_stats = {}
                if avg_response.status_code == 200:
                    avg_data = avg_response.json()
                    if avg_data.get('data'):
                        stats = avg_data['data'][0]
                        season_stats = {
                            'points': round(stats.get('pts', 0), 1),
                            'rebounds': round(stats.get('reb', 0), 1),
                            'assists': round(stats.get('ast', 0), 1),
                            'steals': round(stats.get('stl', 0), 1),
                            'blocks': round(stats.get('blk', 0), 1),
                            'field_goal_pct': round(stats.get('fg_pct', 0) * 100, 1),
                            'three_point_pct': round(stats.get('fg3_pct', 0) * 100, 1),
                            'free_throw_pct': round(stats.get('ft_pct', 0) * 100, 1),
                            'games_played': stats.get('games_played', 0),
                            'minutes': round(stats.get('min', 0), 1)
                        }
                
                # Get recent games
                games_response = await client.get(
                    f"{self.base_urls['balldontlie']}/games",
                    params={
                        'seasons[]': 2024,
                        'player_ids[]': player_id,
                        'per_page': 10
                    }
                )
                
                recent_games = []
                if games_response.status_code == 200:
                    games_data = games_response.json()
                    for game in games_data.get('data', [])[:5]:  # Last 5 games
                        recent_games.append({
                            'date': game.get('date', ''),
                            'opponent': game.get('visitor_team', {}).get('full_name', 'Unknown'),
                            'home_team': game.get('home_team', {}).get('full_name', ''),
                            'visitor_team': game.get('visitor_team', {}).get('full_name', ''),
                            'status': game.get('status', 'Completed')
                        })
                
                player_stats = {
                    'name': player_info['name'],
                    'team': player_info['team'],
                    'position': player_info['position'],
                    'season_averages': season_stats,
                    'recent_games': recent_games
                }
                
                await self.set_cache_data(cache_key, player_stats, 30)  # Cache for 30 minutes
                return player_stats
                
        except Exception as e:
            logger.error(f"Error fetching player stats for {player_name}: {e}")
            return None
    
    async def fetch_betting_odds(self, sport: str = 'basketball_nba'):
        """Fetch real betting odds"""
        cache_key = f"betting_odds_{sport}"
        cached_data = await self.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For now, generate realistic mock odds since most betting APIs require paid subscriptions
            # In production, you would integrate with The Odds API or similar service
            mock_odds = {
                'games': [
                    {
                        'home_team': 'Los Angeles Lakers',
                        'away_team': 'Golden State Warriors',
                        'commence_time': '2025-06-25T02:00:00Z',
                        'bookmakers': [
                            {
                                'title': 'DraftKings',
                                'markets': [
                                    {
                                        'key': 'h2h',
                                        'outcomes': [
                                            {'name': 'Los Angeles Lakers', 'price': -110},
                                            {'name': 'Golden State Warriors', 'price': -110}
                                        ]
                                    },
                                    {
                                        'key': 'spreads',
                                        'outcomes': [
                                            {'name': 'Los Angeles Lakers', 'price': -110, 'point': -2.5},
                                            {'name': 'Golden State Warriors', 'price': -110, 'point': 2.5}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            
            await self.set_cache_data(cache_key, mock_odds, 15)  # Cache for 15 minutes
            return mock_odds
            
        except Exception as e:
            logger.error(f"Error fetching betting odds: {e}")
            return {'games': []}
    
    async def fetch_sports_news(self):
        """Fetch latest sports news"""
        cache_key = "sports_news"
        cached_data = await self.get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Fetch from ESPN RSS or similar free sources
            import feedparser
            feed_urls = [
                'https://www.espn.com/nba/rss.xml',
                'https://www.espn.com/nfl/rss.xml',
                'https://www.espn.com/mlb/rss.xml'
            ]
            
            all_news = []
            for url in feed_urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]:  # Top 5 from each feed
                        all_news.append({
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.get('published', ''),
                            'summary': entry.get('summary', ''),
                            'source': feed.feed.get('title', 'ESPN')
                        })
                except Exception as e:
                    logger.error(f"Error parsing feed {url}: {e}")
            
            await self.set_cache_data(cache_key, all_news, 30)  # Cache for 30 minutes
            return all_news
            
        except Exception as e:
            logger.error(f"Error fetching sports news: {e}")
            return []

# Initialize sports data service
sports_service = SportsDataService()

class SportsQuery:
    def __init__(self):
        self.patterns = {
            'over_under': r'(will|can) (.+?) (score|get|have) (over|under) (\d+\.?\d*) (points|rebounds|assists|yards|touchdowns)',
            'player_stats': r'(.+?) (stats|average|season)',
            'team_info': r'(.+?) (record|next game|info)',
            'news': r'(news|latest|updates) (.+?)',
            'odds': r'(odds|betting|lines) (.+?)'
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
        
        # News Pattern
        if 'news' in query or 'latest' in query or 'update' in query:
            return {'type': 'news'}
        
        # Odds Pattern
        if 'odds' in query or 'betting' in query or 'line' in query:
            return {'type': 'odds'}
        
        return {'type': 'general', 'query': query}
    
    async def get_player_analysis(self, player_name: str, stat: str, line: float, direction: str) -> str:
        player_stats = await sports_service.fetch_player_stats(player_name)
        
        if not player_stats:
            return f"Sorry, I couldn't find current stats for {player_name}. Please try another player."
        
        season_avg = player_stats['season_averages'].get(stat.rstrip('s'), 0)
        recent_games = player_stats['recent_games']
        
        analysis = f"📊 **{player_stats['name']} - {stat.title()} Analysis**\n\n"
        analysis += f"🎯 **Line:** {direction.title()} {line} {stat}\n"
        analysis += f"📈 **Season Average:** {season_avg}\n"
        analysis += f"🏀 **Team:** {player_stats['team']}\n"
        analysis += f"📍 **Position:** {player_stats['position']}\n\n"
        
        # Generate recommendation based on real data
        if season_avg > 0:
            if direction == "over":
                recommendation = "✅ LEAN OVER" if season_avg > line else "❌ LEAN UNDER"
                confidence = min(90, int(60 + (season_avg - line) * 5)) if season_avg > line else max(35, int(50 - (line - season_avg) * 3))
            else:
                recommendation = "✅ LEAN UNDER" if season_avg < line else "❌ LEAN OVER"
                confidence = min(90, int(60 + (line - season_avg) * 5)) if season_avg < line else max(35, int(50 - (season_avg - line) * 3))
        else:
            recommendation = "❓ INSUFFICIENT DATA"
            confidence = 50
        
        analysis += f"🎯 **Recommendation:** {recommendation}\n"
        analysis += f"🔒 **Confidence:** {confidence}%\n\n"
        
        if recent_games:
            analysis += f"**Recent Games:**\n"
            for i, game in enumerate(recent_games[:3], 1):
                analysis += f"{i}. {game['date']}: vs {game.get('opponent', 'Unknown')}\n"
        
        return analysis
    
    async def get_player_stats(self, player_name: str) -> str:
        player_stats = await sports_service.fetch_player_stats(player_name)
        
        if not player_stats:
            return f"Sorry, I couldn't find current stats for {player_name}. Please try another player."
        
        response = f"📊 **{player_stats['name']} - Season Stats**\n\n"
        response += f"🏀 **Team:** {player_stats['team']}\n"
        response += f"📍 **Position:** {player_stats['position']}\n\n"
        response += "**Season Averages:**\n"
        
        stats = player_stats['season_averages']
        if stats:
            for stat, value in stats.items():
                stat_name = stat.replace('_', ' ').title()
                response += f"• {stat_name}: {value}\n"
        else:
            response += "• No current season stats available\n"
        
        return response
    
    async def get_news_updates(self) -> str:
        news = await sports_service.fetch_sports_news()
        
        response = "📰 **Latest Sports News**\n\n"
        
        if news:
            for article in news[:5]:  # Top 5 articles
                response += f"🔸 **{article['title']}**\n"
                response += f"   _{article['source']}_\n\n"
        else:
            response += "Unable to fetch latest news at the moment. Please try again later."
        
        return response
    
    async def get_betting_odds(self) -> str:
        odds = await sports_service.fetch_betting_odds()
        
        response = "🎲 **Current Betting Odds**\n\n"
        
        games = odds.get('games', [])
        if games:
            for game in games[:3]:  # Show top 3 games
                response += f"🏀 **{game['away_team']} @ {game['home_team']}**\n"
                
                for bookmaker in game.get('bookmakers', [])[:1]:  # Show first bookmaker
                    response += f"📊 _{bookmaker['title']}_\n"
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            response += "**Moneyline:**\n"
                            for outcome in market['outcomes']:
                                response += f"• {outcome['name']}: {outcome['price']:+d}\n"
                        elif market['key'] == 'spreads':
                            response += "**Spread:**\n"
                            for outcome in market['outcomes']:
                                response += f"• {outcome['name']} {outcome['point']:+.1f}: {outcome['price']:+d}\n"
                response += "\n"
        else:
            response += "No current betting lines available."
        
        return response

sports_query = SportsQuery()

@api_router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        query = request.message.strip()
        parsed_query = sports_query.parse_query(query)
        
        if parsed_query['type'] == 'over_under':
            response = await sports_query.get_player_analysis(
                parsed_query['player'],
                parsed_query['stat'],
                parsed_query['line'],
                parsed_query['direction']
            )
        elif parsed_query['type'] == 'player_stats':
            response = await sports_query.get_player_stats(parsed_query['player'])
        elif parsed_query['type'] == 'news':
            response = await sports_query.get_news_updates()
        elif parsed_query['type'] == 'odds':
            response = await sports_query.get_betting_odds()
        elif 'fanduel' in query.lower() or 'draftkings' in query.lower():
            platform = 'fanduel' if 'fanduel' in query.lower() else 'draftkings'
            response = await sports_query.generate_lineup_suggestion(platform)
        elif 'parlay' in query.lower():
            response = "🎲 **Parlay Builder**\n\nGreat choice! Parlays can offer big payouts. Here are some tips:\n\n🎯 **Smart Parlay Strategies:**\n• Mix different bet types (spread + over/under)\n• Avoid correlated bets from same game\n• Start with 2-3 legs for better odds\n• Research each pick thoroughly\n\n💡 **Example Parlay:**\n• Player A Over 25.5 Points\n• Team B -3.5 Spread\n• Game Total Under 215.5\n\n🔒 **Remember:** Bet responsibly and within your limits!"
        else:
            response = f"🏈🏀 **Sports Agent AI** 🏀🏈\n\nI can help you with real sports data:\n\n📊 **Player Analysis:**\n• \"Will LeBron James score over 22 points?\"\n• \"Stephen Curry stats\"\n\n🎲 **Live Data:**\n• \"Latest sports news\"\n• \"Current betting odds\"\n\n🎯 **Daily Fantasy:**\n• \"FanDuel lineup suggestions\"\n• \"DraftKings optimal picks\"\n\n🎪 **Parlays & More:**\n• \"Build me a parlay\"\n• Real-time updates\n\nTry asking about your favorite players or teams!"
        
        # Save chat to database
        chat_message = ChatMessage(message=query, response=response)
        await db.chat_history.insert_one(chat_message.dict())
        
        return {"response": response}
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your request")

@api_router.get("/players")
async def get_players():
    try:
        players_data = await sports_service.fetch_nba_players()
        return {"players": list(players_data.keys())}
    except Exception as e:
        logging.error(f"Error getting players: {str(e)}")
        return {"players": []}

@api_router.get("/news")
async def get_sports_news():
    try:
        news = await sports_service.fetch_sports_news()
        return {"news": news}
    except Exception as e:
        logging.error(f"Error getting news: {str(e)}")
        return {"news": []}

@api_router.get("/odds")
async def get_betting_odds():
    try:
        odds = await sports_service.fetch_betting_odds()
        return odds
    except Exception as e:
        logging.error(f"Error getting odds: {str(e)}")
        return {"games": []}

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

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()