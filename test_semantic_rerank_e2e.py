"""
End-to-end test: real Google Maps API + cloud vLLM semantic rerank.

Runs full_pipeline() with preferences so the semantic_rerank step fetches
real Google reviews and re-ranks candidates via the tunneled cloud vLLM.
This is a LOCAL test — no cloud files are touched.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from google_api_based_gis_tools import full_pipeline  # noqa: E402

# Two-person demo scenario, downtown Toronto.
USERS = [
    "Union Station, 65 Front Street West, Toronto, ON",
    "Yonge and Bloor, Toronto, ON",
]
MODES = ["WALK", "BICYCLE"]
PREFERENCES = (
    "Alice is vegan and takes the subway. "
    "Bob hates spicy food and bikes."
)

if __name__ == "__main__":
    print("=== full_pipeline WITH preferences (semantic rerank ON) ===")
    result = full_pipeline(
        user_addresses=USERS,
        transport_modes=MODES,
        place_type="restaurant",
        travel_time="30m",
        requirements={"min_rating": 4.0},
        top_n=4,
        preferences=PREFERENCES,
    )
    print(f"\nstage={result.get('stage')}  reranked={result.get('reranked')}  count={result.get('suggestions_count')}")
    if "error" in result:
        print(f"ERROR: {result['error']} (stage={result.get('stage')})")
        sys.exit(1)
    print("\n--- RERANKED SUGGESTIONS (order + match score) ---")
    for i, p in enumerate(result["suggestions"], 1):
        print(f"{i}. {p['name']} | rating={p.get('rating')} | avg={p.get('avg_travel_time'):.1f}min "
              f"| match={p.get('match_score')} | reviews={len(p.get('reviews', []))}")
        if p.get("match_reason"):
            print(f"     reason: {p['match_reason']}")
    print("\n--- ai_text (first 1200 chars) ---")
    print(result["ai_text"][:1200])
