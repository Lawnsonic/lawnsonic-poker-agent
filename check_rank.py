import os, json
from dotenv import load_dotenv
from examples.arena_client import ArenaClient, DEFAULT_BASE
load_dotenv()
creds = json.loads(open('.arena-credentials').read())
client = ArenaClient(os.environ.get('ARENA_API_BASE', DEFAULT_BASE), api_key=creds['apiKey'])
lb = client.get('/competition/leaderboard?competitionId=seed_poker_eval_s1')
data = lb.get('data', [])
for entry in data[:5]:
    agent = entry.get('agent', {})
    print(f"Rank {entry['rank']}: {agent['name']} -- {entry['totalScore']:.1f} -- quote: {agent['quote']}")
client.close()
