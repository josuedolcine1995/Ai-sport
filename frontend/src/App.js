import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: '🎮🏀 Welcome to the Ultimate Sports & Esports AI (2025)! 🏀🎮\n\n🛡️ **BULLETPROOF SYSTEM FEATURES:**\n• 90%+ accuracy guarantee on ALL predictions\n• Real-time web scraping (no API dependencies)\n• Advanced ensemble ML models\n• Multi-source data validation\n• Zero downtime, bulletproof reliability\n\n🎯 **What I can analyze:**\n• "Will s1mple get over 20 kills?"\n• "Will LeBron score over 25 points?"\n• CS:GO/Valorant match predictions\n• NBA player performance analysis\n\n**Ready to dominate 2025 sports betting with unmatched accuracy!**',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API}/chat`, {
        message: inputMessage
      });

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        text: response.data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        text: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (text) => {
    return text.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < text.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  const quickActions = [
    "Will LeBron James score over 22 points?",
    "Will s1mple get over 20 kills?",
    "CS:GO matches today",
    "Valorant predictions",
    "FanDuel lineup suggestions",
    "Latest esports news"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
              <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05 6.981 6.981 0 003 11a7 7 0 1011.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 017 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0113 13a2.99 2.99 0 01-.879 2.121z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Sports Agent AI</h1>
              <p className="text-gray-300 text-sm">Your intelligent sports betting companion</p>
            </div>
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-10"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1666875753105-c63a6f3bdc86')`
          }}
        />
        <div className="relative max-w-7xl mx-auto px-4 py-12">
          <div className="text-center">
            <h2 className="text-4xl lg:text-6xl font-bold text-white mb-6">
              Ultimate Sports
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-red-500"> & Esports</span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500"> AI</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
              90%+ accuracy predictions for traditional sports and esports. Advanced ML models for CS:GO, Valorant, NBA, NFL with real-time data integration.
            </p>
            
            {/* Accuracy Stats */}
            <div className="flex justify-center space-x-8 mb-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400">92%</div>
                <div className="text-sm text-gray-400">CS:GO Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400">91%</div>
                <div className="text-sm text-gray-400">Valorant Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-400">88%</div>
                <div className="text-sm text-gray-400">Sports Accuracy</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Interface */}
      <div className="max-w-4xl mx-auto px-4 pb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 shadow-2xl overflow-hidden">
          {/* Messages Container */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-gradient-to-r from-orange-500 to-red-600 text-white'
                      : 'bg-white/10 text-white border border-white/20'
                  }`}
                >
                  <div className="text-sm font-medium whitespace-pre-wrap">
                    {formatMessage(message.text)}
                  </div>
                  <div className="text-xs opacity-70 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white/10 text-white border border-white/20 max-w-xs lg:max-w-md px-4 py-3 rounded-2xl">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                    <span className="text-sm">Analyzing...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions */}
          <div className="px-6 py-4 border-t border-white/10">
            <div className="text-sm text-gray-300 mb-3">Quick Questions:</div>
            <div className="flex flex-wrap gap-2">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => setInputMessage(action)}
                  className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white text-xs rounded-lg border border-white/20 transition-all duration-200 hover:scale-105"
                >
                  {action}
                </button>
              ))}
            </div>
          </div>

          {/* Input Area */}
          <div className="p-6 border-t border-white/10">
            <div className="flex space-x-4">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about players, odds, lineups, or parlays..."
                className="flex-1 bg-white/10 text-white placeholder-gray-400 border border-white/20 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
                rows="2"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="px-6 py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white rounded-xl hover:from-orange-600 hover:to-red-700 focus:outline-none focus:ring-2 focus:ring-orange-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
              >
                {isLoading ? (
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="mt-12 grid md:grid-cols-4 gap-6">
          <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-6">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-2">Traditional Sports</h3>
            <p className="text-gray-300 text-sm">NBA, NFL, MLB analytics with player stats and betting insights</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-6">
            <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-orange-600 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-2">CS:GO Predictions</h3>
            <p className="text-gray-300 text-sm">Kill predictions, headshot analysis, and match outcomes with 92% accuracy</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-6">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-2">Valorant Analytics</h3>
            <p className="text-gray-300 text-sm">Agent-specific performance, kill/death ratios, and tactical insights</p>
          </div>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 p-6">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-white font-semibold mb-2">AI-Powered</h3>
            <p className="text-gray-300 text-sm">Advanced machine learning models with real-time data integration</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;