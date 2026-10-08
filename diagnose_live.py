"""Live diagnosis: rapid-poll pending-actions to catch turn windows."""
from dotenv import load_dotenv; load_dotenv()
import sys, os, json, time

sys.path.insert(0, "examples")
creds = json.loads(open(".arena-credentials").read())
from arena_client import ArenaClient

c = ArenaClient(os.environ["ARENA_API_BASE"], api_key=creds["apiKey"])
comp = os.environ["ARENA_COMPETITION_ID"]

print("Rapid-polling pending-actions for 60s (30 polls x 2s)...")
print("comp=" + comp)
for i in range(30):
    r = c.get("/texas/pending-actions?competitionId=" + comp)
    tables = r.get("tables", [])
    lobby = c.get("/texas/lobby?competitionId=" + comp)
    lpos = lobby.get("lobby")
    if lpos:
        lstr = "queue=" + str(lpos["position"]) + "/" + str(lpos["total"])
    else:
        lstr = "lobby=null(SEATED)"
    tag = " <<<<< TURN!" if tables else ""
    print("  [" + str(i).zfill(2) + "] " + lstr + " | pending=" + str(len(tables)) + tag)
    if tables:
        for tbl in tables:
            print("    TABLE id=" + str(tbl["id"]) + " acting=" + str(tbl.get("actingSeatNumber")) + " self=" + str(tbl.get("selfSeatNumber")))
            print("    allowed=" + str(tbl.get("allowedActions")))
    time.sleep(2)

c.close()
