import os, json
from dotenv import load_dotenv
from examples.arena_client import ArenaClient, DEFAULT_BASE
load_dotenv()
creds = json.loads(open('.arena-credentials').read())
client = ArenaClient(os.environ.get('ARENA_API_BASE', DEFAULT_BASE), api_key=creds['apiKey'])
status = client.get('/texas/benchmark/status?competitionId=seed_poker_eval_s1')
match = status.get('match') or {}
print(f"Hands: {match.get('completedHands')}")
print(f"bb/100: {match.get('adjustedBbPer100')}")
print(f"Phase: {match.get('phase')}")
client.close()
